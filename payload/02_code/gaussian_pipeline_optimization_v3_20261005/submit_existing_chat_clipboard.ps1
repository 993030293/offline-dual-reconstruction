param(
    [Parameter(Mandatory=$true)][ValidateSet('a','b')][string]$TaskBranch,
    [string]$TaskManifestPath = 'research/2026-10-05-gaussian-optimization-round3-50/parallel_001/MANIFEST.json',
    [string]$TaskId = '',
    [switch]$ContinueExistingGoal
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class ChatFocus {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr window, out Rectangle rectangle);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extra);
    public struct Rectangle { public int Left, Top, Right, Bottom; }
}
'@
$projectRoot = 'C:/lidar-camera-fusion-mask-robust-downsampling'
$manifestPath = Join-Path $projectRoot $TaskManifestPath
$manifest = Get-Content -LiteralPath $manifestPath -Encoding UTF8 -Raw | ConvertFrom-Json
$selectedTaskId = if ($TaskId) { $TaskId } else { 'batch_' + $TaskBranch + '_001' }
$assignment = $manifest.assignments | Where-Object { $_.task_id -eq $selectedTaskId }
if ($null -eq $assignment) { throw 'Unknown authorized task' }
$authorizedThread = if ($TaskBranch -eq 'a') { '01a10c61-9180-7b33-b15d-d414c526b640' } else { '01a10c61-c330-7502-b21e-2e0c1799e03a' }
if ($assignment.thread_id -ne $authorizedThread -or $selectedTaskId -notmatch ('^batch_' + $TaskBranch + '_\d{3}$')) {
    throw 'The task is outside the explicitly authorized existing chat'
}
$taskPath = Join-Path $projectRoot $assignment.prompt
$taskText = Get-Content -LiteralPath $taskPath -Encoding UTF8 -Raw
if ($taskText.Length -lt $(if ($ContinueExistingGoal) { 100 } else { 1800 })) {
    throw 'Full task text or a bounded existing-task follow-up is unavailable'
}
$correction = if ($selectedTaskId -eq 'batch_a_001') { 'This corrects the earlier incomplete input. ' } else { '' }
$message = $correction + "This is the authorized $selectedTaskId task for existing thread $($assignment.thread_id). Use native goal mode: call get_goal, then create_goal with the following objective if no unfinished goal exists. Do not set a token budget or overwrite another unfinished goal. This message is the complete instruction; execute the full task below and preserve all original constraints. The task chat itself must create and verify its native task goal; a completed prior batch is separate from this new task.`n`nOBJECTIVE:`n$($assignment.objective)`n`n$taskText"
if ($ContinueExistingGoal) {
    $message = "This continues the same authorized $selectedTaskId task in existing thread $($assignment.thread_id). Its waiting condition has changed. Resume the original task in native goal mode, read get_goal and preserve the existing objective and budget. Do not create or replace another unfinished goal. Read and execute the bounded follow-up below; mark the original batch complete only after its real required closure is verified. The parent fifty-optimization goal stays active and separate.`n`n$taskText"
}
# Keep the previous clipboard formats in memory and never log their contents.
$previous = [System.Windows.Forms.Clipboard]::GetDataObject()
$backup = New-Object System.Windows.Forms.DataObject
if ($null -ne $previous) {
    foreach ($format in $previous.GetFormats($false)) {
        $value = $previous.GetData($format,$false)
        if ($value -is [System.IO.MemoryStream]) {
            $value = New-Object System.IO.MemoryStream(,$value.ToArray())
        }
        $backup.SetData($format,$false,$value)
    }
}
$attempted = $false
try {
    [System.Windows.Forms.Clipboard]::SetText($message,[System.Windows.Forms.TextDataFormat]::UnicodeText)
    if ([System.Windows.Forms.Clipboard]::GetText([System.Windows.Forms.TextDataFormat]::UnicodeText) -cne $message) {
        throw 'Unicode clipboard payload differs from the original instruction'
    }
    # Navigate and submit in one call so intervening progress cannot change focus.
    Start-Process -FilePath ('codex://threads/' + $assignment.thread_id)
    Start-Sleep -Milliseconds 1000
    $chatWindow = Get-Process | Where-Object { $_.MainWindowTitle -eq 'ChatGPT' -and $_.MainWindowHandle -ne 0 } | Select-Object -First 1
    if ($null -eq $chatWindow -or [ChatFocus]::GetForegroundWindow() -ne $chatWindow.MainWindowHandle) {
        throw 'The explicitly opened desktop chat is not foreground'
    }
    $rectangle = New-Object ChatFocus+Rectangle
    if (-not [ChatFocus]::GetWindowRect($chatWindow.MainWindowHandle,[ref]$rectangle)) {
        throw 'The existing window rectangle is unavailable'
    }
    $clickX = [int]($rectangle.Left + 0.48 * ($rectangle.Right-$rectangle.Left))
    $clickY = [int]($rectangle.Top + 0.90 * ($rectangle.Bottom-$rectangle.Top))
    [ChatFocus]::SetCursorPos($clickX,$clickY) | Out-Null
    [ChatFocus]::mouse_event(2,0,0,0,[UIntPtr]::Zero)
    [ChatFocus]::mouse_event(4,0,0,0,[UIntPtr]::Zero)
    [System.Windows.Forms.SendKeys]::SendWait('^v')
    Start-Sleep -Milliseconds 250
    [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
    $attempted = $true
    Start-Sleep -Milliseconds 250
}
finally {
    if ($null -eq $previous) {
        [System.Windows.Forms.Clipboard]::Clear()
    } else {
        [System.Windows.Forms.Clipboard]::SetDataObject($backup,$true)
    }
}
[PSCustomObject]@{
    task_id=$selectedTaskId
    target_thread=$assignment.thread_id
    characters=$message.Length
    unicode_payload_checked=$true
    clipboard_restored=$true
    ui_submission_attempted=$attempted
    continuing_same_existing_goal=[bool]$ContinueExistingGoal
    native_delivery_and_goal_verification_pending=$true
} | ConvertTo-Json
