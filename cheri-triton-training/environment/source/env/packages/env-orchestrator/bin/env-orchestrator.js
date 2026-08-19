#!/usr/bin/env node

/**
 * env-orchestrator CLI entry point
 * 
 * This is the executable that gets installed to node_modules/.bin/
 */

import { main } from "../dist/cli/main.js";

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
