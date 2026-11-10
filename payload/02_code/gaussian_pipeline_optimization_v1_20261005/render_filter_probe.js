import * as THREE from "three";
import { PackedSplats, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

// Alpha readback avoids display color-space conversion in the analytic filter comparison.
window.runRenderFilterProbe = async () => {
  const results = [];
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setClearColor(0, 0);
  document.body.appendChild(renderer.domElement);
  try {
    for (let index = 0; index < 12; index++) {
      const width = index < 6 ? 129 : 193, height = index < 6 ? 97 : 145;
      const fx = width * 0.8, fy = height * 1.15, cx = (width + 1) / 2, cy = (height + 1) / 2;
      const depth = index % 2 ? 4 : 2;
      const shape = [[0.2, 0.2], [0.7, 1.2], [3, 2]][index % 3];
      const offX = index % 2 ? 9 : 0, offY = index % 2 ? -7 : 0;
      const camera = new THREE.PerspectiveCamera(50, width / height, 0.02, 100);
      camera.projectionMatrix.set(2*fx/width, 0, 1-2*cx/width, 0,
        0, 2*fy/height, 2*cy/height-1, 0, 0, 0, -(100+.02)/(100-.02), -2*100*.02/(100-.02), 0, 0, -1, 0);
      camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
      renderer.setSize(width, height);
      const packed = new PackedSplats({ lod: false });
      await packed.initialized;
      packed.pushSplat(new THREE.Vector3(offX/fx*depth, -offY/fy*depth, -depth),
        new THREE.Vector3(shape[0]*depth/fx, shape[1]*depth/fy, .001),
        new THREE.Quaternion(), index % 2 ? .4 : .8, new THREE.Color(1, 1, 1));
      const decoded = packed.getSplat(0);
      const record = { index, width, height, fx, fy, cx, cy,
        nominal: { sigma: shape, depth, offX, offY },
        decoded: { center: decoded.center.toArray(), scales: decoded.scales.toArray(),
          quaternion_xyzw: decoded.quaternion.toArray(), opacity: decoded.opacity, color: decoded.color.toArray() },
        variants: [] };
      for (const label of ["default_compensated", "model_classic"]) {
        const scene = new THREE.Scene();
        const spark = new SparkRenderer({ renderer, autoUpdate: false,
          ...(label === "model_classic" ? { preBlurAmount: .3, blurAmount: 0 } : {}) });
        const mesh = new SplatMesh({ splats: packed, lod: false });
        scene.add(spark, mesh);
        await mesh.initialized;
        camera.updateMatrixWorld();
        await spark.update({ scene, camera });
        renderer.render(scene, camera);
        const gl = renderer.getContext(), pixels = new Uint8Array(width*height*4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        const alpha = [];
        for (let y=0; y<height; y++) for (let x=0; x<width; x++) {
          alpha.push(pixels[((height-1-y)*width+x)*4+3]);
        }
        record.variants.push({ label, preBlurAmount: spark.preBlurAmount, blurAmount: spark.blurAmount, alpha });
        scene.remove(mesh, spark);
        spark.dispose();
      }
      results.push(record);
      packed.dispose();
    }
    return { status: "rendered_pending_independent_oracle", results,
      drawingBufferAlpha: renderer.getContext().getContextAttributes().alpha,
      source: "actual_Spark_WebGL_alpha_readback", optimization_count: 0 };
  } finally {
    renderer.dispose();
    renderer.domElement.remove();
  }
};
