# ADSR_Segmenter - Windows installer constants
$script:ADSRSegmenterConfig = @{
    GitHubRepoUrl      = 'https://github.com/LuisMRaimundo/ADSR_Segmenter'
    AppName            = 'ADSR_Segmenter'
    PythonVersion      = '3.11'
    PythonMinMinor     = 10
    PythonMaxMinor     = 12
    PythonInstallerUrl = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'
    BootstrapScript    = 'installers\common\bootstrap.py'
    PortablePythonExe  = 'installers\runtime\windows\python-full\python.exe'
}
