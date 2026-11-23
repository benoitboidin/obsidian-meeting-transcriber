import { readFileSync, writeFileSync } from "fs";
const v = process.env.npm_package_version;
const m = JSON.parse(readFileSync("manifest.json", "utf8"));
m.version = v;
writeFileSync("manifest.json", JSON.stringify(m, null, 2) + "\n");
const vs = JSON.parse(readFileSync("versions.json", "utf8"));
vs[v] = m.minAppVersion;
writeFileSync("versions.json", JSON.stringify(vs, null, 2) + "\n");
