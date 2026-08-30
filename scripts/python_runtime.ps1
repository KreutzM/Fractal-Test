function Resolve-CompatiblePython {
    [CmdletBinding()]
    param()

    $supportedVersionPattern = '^3\.(11|12|13)$'
    $candidatePaths = [System.Collections.Generic.List[string]]::new()

    foreach ($commandName in @('python.exe', 'python')) {
        $commands = @(Get-Command $commandName -All -CommandType Application -ErrorAction SilentlyContinue)
        foreach ($command in $commands) {
            $path = [string]$command.Source
            if ([string]::IsNullOrWhiteSpace($path)) {
                $path = [string]$command.Path
            }

            # The Microsoft Store alias is only a placeholder and cannot create a venv.
            if ([string]::IsNullOrWhiteSpace($path) -or
                $path -like '*\Microsoft\WindowsApps\*' -or
                $candidatePaths.Contains($path)) {
                continue
            }

            $candidatePaths.Add($path)
        }
    }

    foreach ($candidate in $candidatePaths) {
        try {
            $versionOutput = @(& $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
            $version = [string]($versionOutput | Select-Object -First 1)
            $version = $version.Trim()

            if (($LASTEXITCODE -eq 0) -and ($version -match $supportedVersionPattern)) {
                return [PSCustomObject]@{
                    Executable = $candidate
                    Arguments  = @()
                    Version    = $version
                }
            }
        }
        catch {
            # Try the next interpreter candidate.
        }
    }

    $launcherCommands = @(Get-Command py.exe -All -CommandType Application -ErrorAction SilentlyContinue)
    foreach ($launcherCommand in $launcherCommands) {
        $launcherPath = [string]$launcherCommand.Source
        if ([string]::IsNullOrWhiteSpace($launcherPath)) {
            $launcherPath = [string]$launcherCommand.Path
        }
        if ([string]::IsNullOrWhiteSpace($launcherPath)) {
            continue
        }

        foreach ($selector in @('-3.13', '-3.12', '-3.11')) {
            try {
                $versionOutput = @(& $launcherPath $selector -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
                $version = [string]($versionOutput | Select-Object -First 1)
                $version = $version.Trim()

                if (($LASTEXITCODE -eq 0) -and ($version -match $supportedVersionPattern)) {
                    return [PSCustomObject]@{
                        Executable = $launcherPath
                        Arguments  = @($selector)
                        Version    = $version
                    }
                }
            }
            catch {
                # Try the next launcher selector.
            }
        }
    }

    throw "Python 3.11, 3.12 or 3.13 was not found. Install a 64-bit Python from python.org and enable 'Add python.exe to PATH'."
}

function Test-CompatibleVenvPython {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }

    try {
        $versionOutput = @(& $Path -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
        $version = [string]($versionOutput | Select-Object -First 1)
        return (($LASTEXITCODE -eq 0) -and ($version.Trim() -match '^3\.(11|12|13)$'))
    }
    catch {
        return $false
    }
}
