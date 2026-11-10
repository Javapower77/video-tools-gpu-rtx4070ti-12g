[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$ResultPath)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = 'Video Studio - local batch folders'
$form.Size = New-Object System.Drawing.Size(680, 440)
$form.StartPosition = 'CenterScreen'
$form.TopMost = $true
$label = New-Object System.Windows.Forms.Label
$label.Text = 'Drag folders from Explorer into the list. Subfolders are included. Outputs stay beside originals.'
$label.SetBounds(15, 15, 635, 45)
$list = New-Object System.Windows.Forms.ListBox
$list.SetBounds(15, 65, 635, 265)
$list.AllowDrop = $true
$list.SelectionMode = 'MultiExtended'
$list.HorizontalScrollbar = $true

$addFolder = {
    param([string]$Path)
    if (Test-Path -LiteralPath $Path -PathType Container) {
        $full = (Get-Item -LiteralPath $Path).FullName
        if (-not $list.Items.Contains($full)) { [void]$list.Items.Add($full) }
    }
}
$list.add_DragEnter({
    param($sender, $eventArgs)
    if ($eventArgs.Data.GetDataPresent([System.Windows.Forms.DataFormats]::FileDrop)) {
        $eventArgs.Effect = [System.Windows.Forms.DragDropEffects]::Copy
    }
})
$list.add_DragDrop({
    param($sender, $eventArgs)
    foreach ($path in $eventArgs.Data.GetData([System.Windows.Forms.DataFormats]::FileDrop)) {
        & $addFolder $path
    }
})
$add = New-Object System.Windows.Forms.Button
$add.Text = 'Browse...'
$add.SetBounds(15, 345, 110, 35)
$add.add_Click({
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    try {
        if ($dialog.ShowDialog($form) -eq [System.Windows.Forms.DialogResult]::OK) {
            & $addFolder $dialog.SelectedPath
        }
    } finally { $dialog.Dispose() }
})
$remove = New-Object System.Windows.Forms.Button
$remove.Text = 'Remove'
$remove.SetBounds(135, 345, 110, 35)
$remove.add_Click({
    foreach ($item in @($list.SelectedItems)) { $list.Items.Remove($item) }
})
$accept = New-Object System.Windows.Forms.Button
$accept.Text = 'Use folders'
$accept.SetBounds(420, 345, 110, 35)
$accept.add_Click({
    if ($list.Items.Count -eq 0) {
        [void][System.Windows.Forms.MessageBox]::Show($form, 'Add at least one folder.')
        return
    }
    $form.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Close()
})
$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = 'Cancel'
$cancel.SetBounds(540, 345, 110, 35)
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.CancelButton = $cancel
$form.Controls.AddRange(@($label, $list, $add, $remove, $accept, $cancel))
try {
    if ($form.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $paths = @($list.Items | ForEach-Object { [string]$_ })
        $json = ConvertTo-Json -InputObject $paths -Compress
        [System.IO.File]::WriteAllText($ResultPath, $json, (New-Object System.Text.UTF8Encoding($false)))
    }
} finally { $form.Dispose() }