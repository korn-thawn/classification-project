// Scan every package in node_modules and flag licenses outside MIT/BSD/Apache-2.0.
// Exit code 1 if anything GPL/AGPL is found (same rule as tools/check_licenses.py).
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "node_modules");
const ALLOWED = /^(MIT|ISC|0BSD|BSD-2-Clause|BSD-3-Clause|Apache-2\.0|BlueOak-1\.0\.0|CC0-1\.0|Unlicense)$/i;
const FORBIDDEN = /\b(A?GPL|LGPL)/i;

function licenseOf(pkg) {
  const l = pkg.license ?? pkg.licenses;
  if (typeof l === "string") return l;
  if (Array.isArray(l)) return l.map((x) => x.type ?? x).join(" OR ");
  if (l && typeof l === "object") return l.type ?? "UNKNOWN";
  return "UNKNOWN";
}

// SPDX expression is acceptable if any OR-branch is fully made of allowed licenses.
function isAllowed(expr) {
  return expr
    .replace(/[()]/g, "")
    .split(/\s+OR\s+/i)
    .some((branch) => branch.split(/\s+AND\s+/i).every((id) => ALLOWED.test(id.trim())));
}

function* packages(dir) {
  if (!existsSync(dir)) return;
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".")) continue;
    const p = join(dir, name);
    if (name.startsWith("@")) { yield* packages(p); continue; }
    const pj = join(p, "package.json");
    if (existsSync(pj)) {
      yield JSON.parse(readFileSync(pj, "utf8"));
      yield* packages(join(p, "node_modules"));
    }
  }
}

const seen = new Map();
for (const pkg of packages(ROOT)) {
  if (pkg.name && !pkg.private) seen.set(`${pkg.name}@${pkg.version}`, licenseOf(pkg));
}

let forbidden = 0, review = 0;
const md = process.argv.includes("--markdown");
if (md) console.log("| package | license |\n|---|---|");
for (const [id, lic] of [...seen].sort()) {
  let flag = "";
  if (FORBIDDEN.test(lic) && !isAllowed(lic)) { flag = "FORBIDDEN"; forbidden++; }
  else if (!isAllowed(lic)) { flag = "REVIEW"; review++; }
  if (md) console.log(`| ${id} | ${lic} |`);
  else if (flag) console.log(`${flag.padEnd(10)} ${id}  (${lic})`);
}
console.log(`\n${seen.size} packages, ${forbidden} forbidden, ${review} need review`);
process.exit(forbidden ? 1 : 0);
