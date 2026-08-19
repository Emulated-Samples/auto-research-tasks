import assert from "node:assert/strict";
import test from "node:test";

const { BLOCKED_GIT_SUBCOMMANDS, DEFAULT_DISALLOWED_TOOLS } = await import(
  "../dist/agents/ClaudeCodeAgent.js"
);

const bashPatterns = DEFAULT_DISALLOWED_TOOLS.filter(
  (p) => p.startsWith("Bash(") && p.endsWith(")")
).map((p) => p.slice("Bash(".length, -1));

/**
 * Glob semantics for Bash(...) deny patterns: `*` matches any characters
 * (including none), everything else is literal. This mirrors the matching
 * the pattern list already relies on (e.g. "Bash(*--git-dir*hyperfocal/env*)"
 * — see the PATTERN SYNTAX note above DEFAULT_DISALLOWED_TOOLS).
 */
function globToRegex(glob) {
  const escaped = glob
    .split("*")
    .map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
    .join(".*");
  return new RegExp(`^${escaped}$`);
}

function denyPatternsMatching(command) {
  return bashPatterns.filter((p) => globToRegex(p).test(command));
}

test("history-reading git subcommands are blocked bare", () => {
  for (const sub of BLOCKED_GIT_SUBCOMMANDS) {
    const cmd = `git ${sub}`;
    assert.ok(
      denyPatternsMatching(cmd).length > 0,
      `expected '${cmd}' to be denied`
    );
  }
});

test("git global options before the subcommand do not bypass the denylist", () => {
  // The lgb-004 hardening note: `git --no-pager log` and
  // `cd ... && git --no-pager diff HEAD` slipped past the bare
  // `Bash(git log*)` / `Bash(git diff*)` patterns.
  const bypasses = [
    "git --no-pager log",
    "git --no-pager diff HEAD",
    "git -P log -p",
    "git -c core.pager=cat log",
    "git -C /hyperfocal/env log",
    "git -C .. --no-pager show HEAD~3:file.py",
    "git --git-dir=/somewhere/.git show main:environment/x",
    "git --paginate --no-pager rev-list --all",
    "git -c color.ui=false ls-files",
    "git -C /tmp/x cat-file -p HEAD",
  ];
  for (const cmd of bypasses) {
    assert.ok(
      denyPatternsMatching(cmd).length > 0,
      `expected '${cmd}' to be denied`
    );
  }
});

test("workflow git subcommands stay allowed", () => {
  // add/commit/push/pull/remote/status/init/clone/fetch must stay usable
  // (envs that deploy via `git push` depend on it).
  const allowed = [
    "git status",
    "git add file.py",
    "git commit -m 'fix'",
    "git push origin main",
    "git pull",
    "git init",
    "git fetch origin",
    "git clone https://example.com/repo.git",
    "git -C /ws status",
    "git remote -v",
  ];
  for (const cmd of allowed) {
    const hits = denyPatternsMatching(cmd);
    assert.equal(
      hits.length,
      0,
      `expected '${cmd}' to be allowed, denied by: ${hits.join(", ")}`
    );
  }
});
