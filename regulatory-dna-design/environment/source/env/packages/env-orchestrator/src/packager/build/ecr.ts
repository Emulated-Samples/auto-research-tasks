/**
 * ECR interaction for `harbor package`: registry login via the ambient AWS
 * credentials, immutable-tag existence checks, and push/pull helpers.
 *
 * Immutable-tag policy: published tags are content-true (the spec hash
 * covers everything baked), so an existing identical tag is pulled and
 * reused, never overwritten — a rebuild of the same commit is not
 * byte-identical, so a re-push would be rejected anyway, and the pulled
 * image is exactly what consumers will get.
 */

import { spawnSync } from "child_process";

function log(msg: string): void {
  console.log(`[harbor:package] ${msg}`);
}

const ecrLoggedIn = new Set<string>();

/** docker login for ECR registries using the ambient AWS credentials. */
export function ensureRegistryLogin(imageTag: string): void {
  const host = imageTag.split("/")[0];
  const ecrMatch = host.match(/\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$/);
  if (!ecrMatch || ecrLoggedIn.has(host)) return;
  const region = ecrMatch[1];
  log(`  docker login ${host} (via aws ecr get-login-password)...`);
  const rc = spawnSync(
    "bash",
    [
      "-c",
      `aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${host}`,
    ],
    { stdio: ["ignore", "pipe", "pipe"], encoding: "utf-8" }
  );
  if (rc.status !== 0) {
    throw new Error(
      `ECR login for ${host} failed: ${(rc.stderr || rc.stdout || "").trim()}`
    );
  }
  ecrLoggedIn.add(host);
}

/** Whether an ECR image tag already exists (immutable tags are never re-pushed). */
export function ecrTagExists(imageTag: string): boolean {
  const match = imageTag.match(
    /^[^/]+\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com\/(.+):([^:]+)$/
  );
  if (!match) return false;
  const [, region, repo, tag] = match;
  const rc = spawnSync(
    "aws",
    [
      "ecr",
      "describe-images",
      "--region",
      region,
      "--repository-name",
      repo,
      "--image-ids",
      `imageTag=${tag}`,
    ],
    { stdio: ["ignore", "ignore", "ignore"] }
  );
  return rc.status === 0;
}

/**
 * Pull an already-published immutable tag so everything downstream —
 * solution production included — exercises exactly the published bytes.
 */
export function pullPublishedImage(imageTag: string): void {
  ensureRegistryLogin(imageTag);
  const pull = spawnSync("docker", ["pull", "--quiet", imageTag], {
    stdio: "inherit",
  });
  if (pull.status !== 0) {
    throw new Error(`docker pull of published ${imageTag} failed`);
  }
}

/**
 * Push a locally built tag, respecting immutability: if the tag appeared in
 * the registry in the meantime, pull the published bytes over the local
 * build instead of pushing.
 */
export function publishImage(imageTag: string): void {
  ensureRegistryLogin(imageTag);
  if (ecrTagExists(imageTag)) {
    log(
      `  ${imageTag} already published (immutable) — pulling it for ` +
        `validation instead of re-pushing`
    );
    pullPublishedImage(imageTag);
    return;
  }
  log(`  docker push ${imageTag}...`);
  const push = spawnSync("docker", ["push", imageTag], { stdio: "inherit" });
  if (push.status !== 0) {
    throw new Error(
      `docker push failed for ${imageTag} — if the repository does ` +
        `not exist yet it must be created first (builders cannot ` +
        `create ECR repositories by design)`
    );
  }
}

/** Whether the docker daemon has an image for this tag locally. */
export function localImageExists(imageTag: string): boolean {
  const rc = spawnSync("docker", ["image", "inspect", imageTag], {
    stdio: ["ignore", "ignore", "ignore"],
  });
  return rc.status === 0;
}
