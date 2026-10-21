param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $TaskArguments = @('verify')
)

& wsl.exe --distribution Ubuntu-24.04 --exec bash /home/bogger/10weekplan/week3/docker/pytorch3d/p3d.sh @TaskArguments
exit $LASTEXITCODE
