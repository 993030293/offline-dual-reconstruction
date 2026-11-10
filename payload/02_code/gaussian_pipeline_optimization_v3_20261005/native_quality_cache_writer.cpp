// Own one original durable cache-file lifecycle with no retained native handles.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <cstdint>

using AuditRename = int (__cdecl*)();

extern "C" __declspec(dllexport) int quality_cache_writer_abi() noexcept { return 1; }

static int publish(
    const wchar_t* temporary, const wchar_t* target, const unsigned char* body,
    std::uint64_t length, AuditRename audit, std::uint64_t* result, std::uint64_t count,
    unsigned fault, DWORD fault_error) noexcept {
    if (!temporary || !target || !body || !audit || !result || count != 5 || !length || length > 16384) return 8;
    for (int index = 0; index < 5; ++index) result[index] = 0;
    HANDLE handle = CreateFileW(temporary, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                               nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        result[0] = GetLastError();
        return 1;
    }
    result[1] = 1;
    std::uint64_t total = 0;
    while (total < length) {
        DWORD written = 0;
        const DWORD requested = static_cast<DWORD>(fault == 1 ? length / 2 : length - total);
        BOOL succeeded = WriteFile(handle, body + total, requested, &written, nullptr);
        if (fault == 1 && succeeded) {
            total += written;
            written = 0;
            succeeded = FALSE;
            SetLastError(fault_error);
        }
        if (!succeeded || !written) {
            result[0] = succeeded ? ERROR_WRITE_FAULT : GetLastError();
            result[4] = total;
            result[3] = CloseHandle(handle) != 0;
            return 2;
        }
        total += written;
    }
    result[4] = total;
    const BOOL flushed = fault == 2 ? (SetLastError(fault_error), FALSE) : FlushFileBuffers(handle);
    if (!flushed) {
        result[0] = GetLastError();
        result[3] = CloseHandle(handle) != 0;
        return 3;
    }
    result[2] = 1;
    if (!CloseHandle(handle)) {
        result[0] = GetLastError();
        return 4;
    }
    result[3] = 1;
    if (!audit()) return 6;
    if (!MoveFileExW(temporary, target, MOVEFILE_REPLACE_EXISTING)) {
        result[0] = GetLastError();
        return 5;
    }
    return 0;
}

extern "C" __declspec(dllexport) int quality_cache_publish(
    const wchar_t* temporary, const wchar_t* target, const unsigned char* body,
    std::uint64_t length, AuditRename audit, std::uint64_t* result, std::uint64_t count) noexcept {
    return publish(temporary, target, body, length, audit, result, count, 0, 0);
}

// Exercise the same failure branches without filling or damaging a real disk.
extern "C" __declspec(dllexport) int quality_cache_publish_fault(
    const wchar_t* temporary, const wchar_t* target, const unsigned char* body,
    std::uint64_t length, AuditRename audit, std::uint64_t* result, std::uint64_t count,
    unsigned fault, DWORD fault_error) noexcept {
    if (fault < 1 || fault > 2 || !fault_error) return 8;
    return publish(temporary, target, body, length, audit, result, count, fault, fault_error);
}
