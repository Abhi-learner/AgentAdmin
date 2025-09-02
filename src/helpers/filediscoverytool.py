class FileDiscoveryTool:
    def __init__(self, hostname: str, username: str, key_file: str = None, password: str = None,
                 port: int = 22, threshold_mb: int = 100):
        self.hostname = hostname
        self.username = username
        self.key_file = key_file
        self.password = password
        self.port = port
        self.threshold_mb = threshold_mb

    def _run_remote_command(self, command: str) -> str:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh.connect(
            hostname=self.hostname,
            port=self.port,                # 👈 custom port here
            username=self.username,
            key_filename=self.key_file,
            password=self.password
        )

        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode()
        ssh.close()
        return output

    def discover_large_files(self, state: Dict[str, Any]) -> Dict[str, Any]:
        fs = state.get("disk_alert_info", {}).get("filesystem", "/var/log")
        command = f"find {fs} -type f -size +{self.threshold_mb}M -exec ls -lh {{}} \\; | awk '{{print $9, $5}}'"

        print(f"📡 Running remote discovery on {self.hostname}:{self.port}{fs}")
        output = self._run_remote_command(command)

        files: List[Dict[str, Any]] = []
        for line in output.strip().splitlines():
            try:
                path, size = line.split()
                files.append({"name": path, "size": size})
            except ValueError:
                continue

        state["candidate_files"] = files
        print("📂 Discovered large files:", files)
        return state