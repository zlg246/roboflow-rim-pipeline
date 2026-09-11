# ============================================================
# setup_scheduler.ps1
# Run ONCE as Administrator to register the weekly task.
#
# Right-click PowerShell -> "Run as Administrator", then:
#   cd C:\ai-train\roboflow_rim_pipeline
#   .\setup_scheduler.ps1
#
# To change the schedule, edit the variables below and re-run.
# ============================================================

# ── CONFIGURE THESE ──────────────────────────────────────────
$RunDay   = "Wednesday"   # Monday / Tuesday / Wednesday / Thursday / Friday / Saturday / Sunday
$RunTime  = "17:00"       # 24h format: "17:00" = 5pm  "09:00" = 9am  "08:30" = 8:30am
$MaxHours = 72            # Maximum hours the task is allowed to run before being killed
# ─────────────────────────────────────────────────────────────

$TaskName    = "RimPipeline"
$ProjectDir  = "C:\ai-train\roboflow_rim_pipeline"
$BatchScript = "$ProjectDir\run_rim_pipeline.bat"
$Description = "Weekly rim pipeline (all dealerships) - runs every $RunDay at $RunTime (max ${MaxHours}h)"

# ── Validate inputs ───────────────────────────────────────────
$ValidDays = @("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
if ($RunDay -notin $ValidDays) {
    Write-Host "Invalid RunDay '$RunDay'. Must be one of: $($ValidDays -join ', ')"
    exit 1
}
if ($RunTime -notmatch '^\d{2}:\d{2}$') {
    Write-Host "Invalid RunTime '$RunTime'. Use 24h format like '17:00' or '09:30'"
    exit 1
}
if ($MaxHours -lt 1 -or $MaxHours -gt 72) {
    Write-Host "Invalid MaxHours '$MaxHours'. Must be between 1 and 72."
    exit 1
}

Write-Host ""
Write-Host "  Task name : $TaskName"
Write-Host "  Project   : $ProjectDir"
Write-Host "  Script    : $BatchScript"
Write-Host "  Schedule  : Every $RunDay at $RunTime"
Write-Host "  Max hours : $MaxHours hour(s)"
Write-Host ""

# ── Remove existing task if it exists ────────────────────────
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task: $TaskName"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# ── Define the action (what to run) ──────────────────────────
$Action = New-ScheduledTaskAction `
    -Execute          "cmd.exe" `
    -Argument         "/c `"$BatchScript`"" `
    -WorkingDirectory $ProjectDir

# ── Define the trigger ────────────────────────────────────────
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek $RunDay `
    -At $RunTime

# ── Define settings ───────────────────────────────────────────
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit        (New-TimeSpan -Hours $MaxHours) `
    -StartWhenAvailable        `
    -RunOnlyIfNetworkAvailable `
    -WakeToRun                 `
    -MultipleInstances         IgnoreNew

# ── Define principal (run as current user, non-interactive) ──
# S4U = Service For User: runs whether user is logged on or not,
# no console window is created.
$Principal = New-ScheduledTaskPrincipal `
    -UserId    "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U `
    -RunLevel  Highest

# ── Register the task ─────────────────────────────────────────
Register-ScheduledTask `
    -TaskName    $TaskName `
    -Action      $Action `
    -Trigger     $Trigger `
    -Settings    $Settings `
    -Principal   $Principal `
    -Description $Description `
    -Force

Write-Host ""
Write-Host "Task '$TaskName' registered successfully."
Write-Host "   Runs every $RunDay at $RunTime"
Write-Host "   Script    : $BatchScript"
Write-Host "   Max hours : $MaxHours"
Write-Host "   Logs      : $ProjectDir\logs\scheduler\"
Write-Host ""
Write-Host "To verify  : Task Scheduler -> Task Scheduler Library -> $TaskName"
Write-Host "To test now: Right-click the task -> Run"
Write-Host ""
Write-Host "To change schedule: edit the variables at the top of this file and re-run."
