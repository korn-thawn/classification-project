// Run the Node API (with --watch) and the Vite dev server together; Ctrl+C stops both.
import { spawn } from "node:child_process";

const procs = [
  ["server", ["run", "dev", "-w", "server"]],
  ["client", ["run", "dev", "-w", "client"]],
].map(([name, args]) => {
  const p = spawn("npm", args, { stdio: "inherit", shell: true });
  p.on("exit", (code) => console.log(`[${name}] exited with code ${code}`));
  return p;
});

process.on("SIGINT", () => {
  procs.forEach((p) => p.kill());
  process.exit(0);
});
