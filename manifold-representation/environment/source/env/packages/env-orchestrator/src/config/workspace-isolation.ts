/**
 * Workspace Isolation
 *
 * Manages Linux user isolation for the agent process.
 * Creates a restricted user and sets up filesystem permissions.
 */

import { execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

const AGENT_USER = "hyperfocal-agent";
const LOGS_DIR = "/hyperfocal/logs";
const AGENT_HOME = `/home/${AGENT_USER}`;

/**
 * Ensure the agent user exists, creating if necessary
 */
export function ensureAgentUser(): void {
  try {
    execSync(`id ${AGENT_USER}`, { stdio: "ignore" });
    console.log(`✅ User ${AGENT_USER} exists`);
  } catch {
    console.log(`Creating ${AGENT_USER} user...`);
    execSync(
      `useradd --system --no-create-home --shell /bin/false ${AGENT_USER}`
    );
    console.log(`✅ Created ${AGENT_USER} user`);
  }

  // The user is created as a system account without a home directory, but
  // CLI agents still need a writable HOME for caches and runtime config
  // (OpenCode: ~/.cache and ~/.local/share/opencode).
  fs.mkdirSync(AGENT_HOME, { recursive: true });
  execSync(`chown -R ${AGENT_USER}:${AGENT_USER} ${AGENT_HOME}`);
  execSync(`chmod 750 ${AGENT_HOME}`);
}

/**
 * Set up workspace permissions for agent isolation
 *
 * @param workspacePath - Path to the workspace directory
 * @param dryRun - If true, only log what would be done
 */
export function setupWorkspacePermissions(
  workspacePath: string,
  dryRun = false
): void {
  const log = (msg: string) =>
    console.log(dryRun ? `[dry-run] ${msg}` : msg);

  // Always reconcile ownership and permissions for the full workspace tree.
  // Individual candidate directories can be recreated as root during setup,
  // so checking only the workspace root can leave nested paths unwritable.

  if (dryRun) {
    log(`Would create workspace: ${workspacePath}`);
    log(`Would create logs directory: ${LOGS_DIR}`);
    log(`Would set ownership: ${AGENT_USER}:${AGENT_USER}`);
    log(`Would set permissions: 750`);
    return;
  }

  // Ensure workspace exists
  if (!fs.existsSync(workspacePath)) {
    fs.mkdirSync(workspacePath, { recursive: true });
    log(`Created workspace directory: ${workspacePath}`);
  }

  // Create shared logs directory (world-writable so agent can write too)
  if (!fs.existsSync(LOGS_DIR)) {
    fs.mkdirSync(LOGS_DIR, { recursive: true });
    execSync(`chmod 777 ${LOGS_DIR}`);
    log(`Created logs directory: ${LOGS_DIR}`);
  }

  // Set ownership using shell command (fs.chownSync requires numeric uid/gid)
  execSync(`chown -R ${AGENT_USER}:${AGENT_USER} ${workspacePath}`);
  log(`Set ownership of ${workspacePath} to ${AGENT_USER}`);

  // Set permissions (owner: rwx, group: rx, other: none)
  execSync(`chmod -R 750 ${workspacePath}`);
  log(`Set permissions of ${workspacePath} to 750`);

  console.log(`✅ Workspace ${workspacePath} owned by ${AGENT_USER}`);
}

/**
 * Lock down sensitive directories so agent cannot read them
 *
 * The agent NEEDS to read packages/ to run the orchestrator code.
 * We lock down everything else that could leak test info or architecture:
 * - environment/    → tests, gold-state, setup code (prevent cheating)
 * - .git/           → git history (prevent repo inspection)
 * - hyperfocal.yaml → config with paths and settings
 * - CLAUDE.md       → project structure docs (Claude Code auto-loads from parent dirs)
 * - AGENTS.md       → same as CLAUDE.md
 * - .claude/        → project settings, agents, commands (Claude Code auto-loads)
 *
 * IMPORTANT: We remove BOTH group and other permissions (go-rwx), not just
 * other (o-rwx). The agent user may share a group with these root-owned files
 * (historically it was added to the root group); stripping group access too
 * guarantees the unprivileged agent cannot read them via any group membership.
 *
 * NOTE: CLAUDE.md / AGENTS.md / .claude/ are deleted from problem branches
 * (Phase 1 cleanup). This lockdown is defense-in-depth in case they reappear
 * after a git checkout or branch switch.
 *
 * @param envRoot - The environment root (e.g., /hyperfocal/env)
 * @param dryRun - If true, only log what would be done
 */
export function lockdownSensitiveDirectories(envRoot: string, dryRun = false): void {
  const log = (msg: string) =>
    console.log(dryRun ? `[dry-run] ${msg}` : msg);

  const hyperfocalRoot = path.dirname(envRoot);

  // Directories/files that should NOT be accessible by the agent.
  // Keep this list explicit and focused on known leakage vectors.
  const sensitivePaths = [
    path.join(envRoot, "hyperfocal.yaml"),                    // Config with internal paths/settings
    path.join(envRoot, "environment"),                        // Tests, gold-state, setup code
    path.join(envRoot, ".git"),                               // Git metadata/history
    path.join(envRoot, "CLAUDE.md"),                          // Auto-loaded docs from parent dirs
    path.join(envRoot, "AGENTS.md"),                          // Same as CLAUDE.md
    path.join(envRoot, ".claude"),                            // Project-level Claude settings
    path.join(envRoot, "packages/migration-env-builder"),     // Migration internals
    path.join(hyperfocalRoot, "migration-env-builder"),       // Alternate migration path
    path.join(hyperfocalRoot, "tmp"),                         // Temp leak area
    path.join(hyperfocalRoot, "z_archive"),                   // Archived runs/history
  ];

  for (const fullPath of sensitivePaths) {
    if (!fs.existsSync(fullPath)) {
      continue;
    }

    if (dryRun) {
      log(`Would lock down ${fullPath} (remove group+other access)`);
      continue;
    }

    try {
      // Remove BOTH group and other permissions (not just other): the agent may
      // share a group with these root-owned files, so o-rwx alone is insufficient.
      if (fs.statSync(fullPath).isDirectory()) {
        execSync(`chmod -R go-rwx ${fullPath}`);
      } else {
        execSync(`chmod go-rwx ${fullPath}`);
      }
      log(`Locked down ${fullPath} (removed group+other access)`);
    } catch (error) {
      console.error(`Failed to lock down ${fullPath}:`, error);
    }
  }

  // Platform-level paths that must be reachable ONLY by root, even under
  // linux-user isolation. These hold cross-run or verifier-critical state that
  // NO agent (root or unprivileged) should read or write:
  //   - /root/.claude/projects     cross-run Claude memory/logs
  //   - /var/lib/hyperfocal-grader  the platform convention for grader/probe/
  //                                 gold runtime state (probe JSONL, the
  //                                 source-state baseline the grader diffs
  //                                 against). This is the reward-hacking crown
  //                                 jewel: read it and you learn the thresholds;
  //                                 write it and you can fake a clean result.
  //                                 Kernel-enforcing it here (chown root + 700)
  //                                 is the generic, env-AGNOSTIC guard — any env
  //                                 that parks verifier state under this path is
  //                                 protected without per-env config. The
  //                                 hyperfocal.yaml disallowedTools patterns are
  //                                 only defense-in-depth (string-matched, and
  //                                 bypassable by a root agent).
  // Each missing path is created first so lock-down can't be bypassed by the
  // agent creating it later, then chown root:root + chmod 700 (dir) / 600 (file).
  const rootOnlyPaths = [
    "/root/.claude/projects",
    "/var/lib/hyperfocal-grader",
  ];

  for (const fullPath of rootOnlyPaths) {
    if (!fs.existsSync(fullPath)) {
      if (dryRun) {
        log(`Would create + lock down ${fullPath} (root-only)`);
        continue;
      }
      try {
        // Ensure the path exists so lock-down cannot be bypassed by creating it later.
        fs.mkdirSync(fullPath, { recursive: true });
        log(`Created ${fullPath} for explicit lock-down`);
      } catch (error) {
        console.error(`Failed to create ${fullPath} for lock-down:`, error);
        continue;
      }
    }

    if (dryRun) {
      log(`Would lock down ${fullPath} (root-only: chown root:root + 700)`);
      continue;
    }

    try {
      execSync(`chown -R root:root ${fullPath}`);
      if (fs.statSync(fullPath).isDirectory()) {
        execSync(`chmod -R go-rwx ${fullPath}`);
        execSync(`chmod 700 ${fullPath}`);
      } else {
        execSync(`chmod 600 ${fullPath}`);
      }
      log(`Locked down ${fullPath} (root-only)`);
    } catch (error) {
      console.error(`Failed to lock down ${fullPath}:`, error);
    }
  }

  console.log(`✅ Sensitive directories locked down`);
}

/**
 * Get the agent user name
 */
export function getAgentUser(): string {
  return AGENT_USER;
}

/**
 * Get the shared logs directory path
 */
export function getLogsDir(): string {
  return LOGS_DIR;
}

/**
 * Check if workspace isolation is set up correctly
 */
export function isIsolationSetUp(workspacePath: string): boolean {
  try {
    // Check user exists
    execSync(`id ${AGENT_USER}`, { stdio: "ignore" });

    // Check workspace ownership
    const stat = fs.statSync(workspacePath);
    const agentUid = parseInt(
      execSync(`id -u ${AGENT_USER}`).toString().trim()
    );
    return stat.uid === agentUid;
  } catch {
    return false;
  }
}

/**
 * Provision Claude Code CLI access for the unprivileged agent user (linux-user
 * mode only).
 *
 * Claude Code authenticates via OAuth at $HOME/.claude/.credentials.json and
 * reads $HOME/.claude.json for config/state. Under linux-user isolation the
 * agent runs with HOME=/home/hyperfocal-agent, so we COPY just those files out
 * of root's home into the agent's home (owned by the agent, mode 0600). We do
 * NOT open /root to the agent, and the agent is NOT placed in the root group.
 *
 * Why copy instead of the previous approach (`usermod -aG root` +
 * `chmod -R g+rwX /root`):
 * - Sound posture: /root stays 700 root-only, so the agent cannot reach
 *   anything else under root's home (SSH keys, shell history, other secrets) or
 *   any other root:root group-readable file on the box. The kernel enforces it
 *   — there is no string pattern to bypass. This is the whole point of running
 *   the agent off root.
 * - The copy carries the OAuth refresh token, so the CLI refreshes its own
 *   token within the agent's HOME for the rollout's duration.
 * - /root/.claude/projects (cross-run memory/logs) is never copied; it is also
 *   hardened root-only by lockdownSensitiveDirectories().
 *
 * npm/pip caches are intentionally NOT shared. Under linux-user the agent
 * builds its own caches under its HOME on first use — correctness over a cold
 * start, on what are throwaway single-rollout VMs. (If cold-start cost ever
 * matters, grant a narrow read-only `g+rX` on just the cache trees rather than
 * reopening /root.)
 *
 * /root/.claude.json carries the CLI's global state (onboarding-complete +
 * OAuth-account fields it needs to run headlessly) but ALSO a `projects` map
 * with per-directory history / MCP approvals / trust for whatever dirs root
 * worked in. We copy it with `projects` emptied so that cross-context state
 * never leaks into the agent's HOME (see the strip in setupClaudeCredentials).
 */
export function setupClaudeCredentials(): void {
  console.log("Provisioning Claude Code CLI access for agent user...");

  const claudeDir = path.join(AGENT_HOME, ".claude");
  fs.mkdirSync(claudeDir, { recursive: true });

  // Copy only the specific auth/config files the CLI needs into the agent's
  // HOME. Sources are root-provisioned at boot; we never copy
  // /root/.claude/projects (cross-run memory). For .claude.json we strip the
  // `projects` map (per-directory history / MCP approvals / trust) so root's
  // other-directory state can't leak into the agent's HOME.
  const credCopies: Array<{ src: string; dst: string; required?: boolean; stripProjects?: boolean }> = [
    { src: "/root/.claude/.credentials.json", dst: path.join(claudeDir, ".credentials.json"), required: true },
    { src: "/root/.claude/settings.json", dst: path.join(claudeDir, "settings.json") },
    { src: "/root/.claude.json", dst: path.join(AGENT_HOME, ".claude.json"), stripProjects: true },
  ];

  let provisionedCreds = false;
  for (const { src, dst, required, stripProjects } of credCopies) {
    if (!fs.existsSync(src)) {
      if (required) {
        console.warn(
          `  ⚠️  Required Claude credential not found at ${src} — the agent may ` +
          `fail to authenticate. Ensure the rollout host provisioned it.`
        );
      }
      continue;
    }
    try {
      if (stripProjects) {
        // Empty `projects` to avoid cross-context leakage, keeping every other
        // (onboarding/account) field the CLI needs. Fall back to a verbatim
        // copy on any parse failure rather than risk breaking auth.
        let wroteStripped = false;
        try {
          const cfg = JSON.parse(fs.readFileSync(src, "utf-8"));
          if (cfg && typeof cfg === "object") {
            cfg.projects = {};
            fs.writeFileSync(dst, JSON.stringify(cfg));
            wroteStripped = true;
          }
        } catch (parseErr) {
          console.warn(`  ⚠️  Could not parse ${src} to strip projects (${parseErr}); copying verbatim`);
        }
        if (!wroteStripped) fs.copyFileSync(src, dst);
      } else {
        fs.copyFileSync(src, dst);
      }
      fs.chmodSync(dst, 0o600);
      if (required) provisionedCreds = true;
      console.log(`  Provisioned ${dst}${stripProjects ? " (projects stripped)" : ""}`);
    } catch (error) {
      console.error(`  Failed to provision ${dst} from ${src}:`, error);
      if (required) throw error;
    }
  }

  // Own the copied auth/config tree as the agent so the unprivileged CLI can
  // read (and rewrite, on token refresh) its own files.
  execSync(`chown -R ${AGENT_USER}:${AGENT_USER} ${claudeDir}`);
  const agentClaudeJson = path.join(AGENT_HOME, ".claude.json");
  if (fs.existsSync(agentClaudeJson)) {
    execSync(`chown ${AGENT_USER}:${AGENT_USER} ${agentClaudeJson}`);
  }

  if (!provisionedCreds) {
    console.warn("  ⚠️  Claude OAuth credentials were not provisioned to the agent HOME.");
  }

  // Copy the CLI binary to /usr/local/bin (world-executable) if not present, so
  // the agent can run it without any access to /root.
  const systemCli = "/usr/local/bin/claude";
  const userCli = "/root/.local/bin/claude";
  if (!fs.existsSync(systemCli)) {
    if (fs.existsSync(userCli)) {
      try {
        // Resolve symlink before copying
        const resolvedPath = fs.realpathSync(userCli);
        execSync(`cp ${resolvedPath} ${systemCli}`);
        execSync(`chmod 755 ${systemCli}`);
        console.log("  Copied Claude CLI to /usr/local/bin/claude");
      } catch (error) {
        console.error("  Failed to copy Claude CLI:", error);
        throw error;
      }
    } else {
      console.warn("  ⚠️  Claude CLI not found at /root/.local/bin/claude");
      console.warn("     Install with: curl -fsSL https://claude.ai/install.sh | bash");
    }
  } else {
    console.log("  Claude CLI already at /usr/local/bin/claude");
  }

  console.log("✅ Claude Code CLI access configured for agent user");
}

/**
 * Check if Claude Code CLI is properly set up for the agent.
 *
 * Verifies the system CLI binary and that OAuth credentials were provisioned
 * into the AGENT's HOME (not /root) — matching setupClaudeCredentials(), which
 * copies creds to the agent's home and no longer relies on root-group access.
 */
export function isClaudeSetUp(): boolean {
  try {
    // Check CLI exists (world-executable copy)
    if (!fs.existsSync("/usr/local/bin/claude")) {
      return false;
    }

    // Check OAuth credentials were copied into the agent's HOME
    if (!fs.existsSync(path.join(AGENT_HOME, ".claude", ".credentials.json"))) {
      return false;
    }

    return true;
  } catch {
    return false;
  }
}
