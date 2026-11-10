"""Stage image resolution while keeping calibrated cameras and radius units explicit."""

from dataclasses import asdict
import time

from full_intrinsics import Pinhole, apply_to_camera


class ProgressiveResolution:
    """Restore original tensors after a bounded area-averaged warmup."""

    def __init__(self, cameras, until_iteration):
        import torch
        import torch.nn.functional as functional

        if until_iteration < 1:
            raise ValueError("A progressive schedule requires a positive warmup")
        self.until = until_iteration
        self.phase = "coarse"
        self.samples = {"coarse": 0, "full": 0}
        self.records = []
        started = time.monotonic()
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for camera in cameras:
                intrinsic = Pinhole(**camera.calibrated_intrinsics)
                if (intrinsic.width % 2 or intrinsic.height % 2 or min(intrinsic.width, intrinsic.height) < 22):
                    raise ValueError("The registered twofold pyramid requires even dimensions and SSIM support")
                if getattr(camera, "depth_reliable", False) or getattr(camera, "invdepthmap", None) is not None:
                    raise ValueError("Depth supervision requires a separately verified multiscale contract")
                image, mask = camera.original_image, camera.alpha_mask
                if image.shape != (3, intrinsic.height, intrinsic.width):
                    raise ValueError("The full image and calibration disagree")
                if mask is not None and (mask.shape != (1, intrinsic.height, intrinsic.width) or not torch.all(mask == 1).item()):
                    raise ValueError("The first pyramid experiment supports only opaque source images")
                self.records.append({"camera": camera, "intrinsic": intrinsic, "image": image, "mask": mask,
                    "coarse_image": functional.avg_pool2d(image, kernel_size=2, stride=2),
                    "coarse_mask": functional.avg_pool2d(mask, kernel_size=2, stride=2) if mask is not None else None,
                    "world_view": camera.world_view_transform.clone()})
        finally:
            torch.set_num_threads(threads)
        if not self.records or len({id(row["camera"]) for row in self.records}) != len(self.records):
            raise ValueError("The camera list is empty or contains repeated objects")
        self.by_object = {id(row["camera"]): row for row in self.records}
        self._apply("coarse")
        self.preparation_seconds = time.monotonic()-started
        self.transition_seconds = 0.

    def _apply(self, phase):
        import torch

        for row in self.records:
            camera, intrinsic = row["camera"], row["intrinsic"]
            coarse = phase == "coarse"
            camera.original_image = row["coarse_image"] if coarse else row["image"]
            camera.alpha_mask = row["coarse_mask"] if coarse else row["mask"]
            camera.image_height, camera.image_width = camera.original_image.shape[-2:]
            scaled = intrinsic.resized(camera.image_width, camera.image_height)
            apply_to_camera(camera, scaled)
            if not torch.equal(camera.world_view_transform, row["world_view"]):
                raise ValueError("Resolution staging changed a metric camera pose")
        self.phase = phase

    def wrap_renderer(self, render):
        def staged(camera, *args, **kwargs):
            import torch

            row = self.by_object.get(id(camera))
            if row is None:
                raise ValueError("An unregistered camera entered the progressive training renderer")
            result = render(camera, *args, **kwargs)
            if tuple(result["render"].shape) != tuple(camera.original_image.shape):
                raise ValueError("Rendered and reference image sizes disagree")
            # Native pruning accumulates radii in the fixed full-resolution pixel unit.
            scale = row["intrinsic"].width / camera.image_width
            if scale != row["intrinsic"].height / camera.image_height or scale not in (1., 2.):
                raise ValueError("The radius conversion requires the declared isotropic scale")
            if scale != 1:
                result = dict(result)
                result["radii"] = result["radii"].to(torch.float32) * scale
            self.samples[self.phase] += 1
            return result
        return staged

    def after_iteration(self, iteration):
        if sum(self.samples.values()) != iteration:
            raise ValueError("The progressive schedule lost its iteration/render correspondence")
        if iteration == self.until:
            started = time.monotonic()
            self._apply("full")
            self.transition_seconds = time.monotonic()-started
        elif iteration > self.until and self.phase != "full":
            raise ValueError("The full-resolution transition was missed")

    def finish(self, iterations):
        import torch

        if self.phase != "full" or self.samples != {"coarse": self.until, "full": iterations-self.until}:
            raise ValueError("The run did not execute its exact declared resolution schedule")
        for row in self.records:
            camera = row["camera"]
            if (camera.original_image is not row["image"] or camera.alpha_mask is not row["mask"]
                    or camera.calibrated_intrinsics != asdict(row["intrinsic"])
                    or not torch.equal(camera.world_view_transform, row["world_view"])):
                raise ValueError("The original image, mask, calibration, or metric pose was not restored")
        return {"until_iteration": self.until, "factor": 2, "samples": self.samples,
                "cameras": len(self.records), "radius_unit": "original_full_resolution_pixels",
                "preparation_seconds": self.preparation_seconds, "transition_seconds": self.transition_seconds,
                "original_tensor_references_and_calibration_restored": True}
