#!/bin/bash
# NeurHL LIVE: publish forecast files through the bot clone (PLAN_NeurHL4 LIVE).
#
# usage: neurhl/live/publish.sh [BOT_CLONE] [FILE_OR_DIR ...]
#
#   BOT_CLONE    a clone of github.com/ph05/neurhl on main
#                (default /Users/ph/Development/nhl-2026-2027-models-live; "" or "-" = default)
#   FILE_OR_DIR  paths relative to this repo's root (or absolute paths inside it). Each is
#                copied to the same relative path in the bot clone. Only paths under
#                neurhl/output/live/, data/manual/, data/raw/rosters/ and docs/ are accepted.
#
# Steps: git pull --rebase; copy; git add ONLY the four trees above; commit with a plain
# message (PUBLISH_MSG, default "live: update <UTC time>"; no trailers of any kind); push to
# main with 3 retries (pull --rebase between tries); verify with `git ls-remote` that the
# commit is on origin/main. The last stdout line is the published commit sha.
#
# Exit codes: 0 ok (also when there was nothing new to commit), 2 usage/bad path,
#             3 bot clone not usable or pull failed, 4 commit refused, 5 push failed,
#             6 commit not on origin/main after push.
set -uo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_BOT="/Users/ph/Development/nhl-2026-2027-models-live"
BOT="${1:-}"
[ "$#" -gt 0 ] && shift
if [ -z "$BOT" ] || [ "$BOT" = "-" ]; then BOT="$DEFAULT_BOT"; fi
ALLOW=(neurhl/output/live data/manual data/raw/rosters docs)
MSG="${PUBLISH_MSG:-live: update $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

log() { echo "[publish $(date -u +%H:%M:%SZ)] $*" >&2; }
die() { local code=$1; shift; log "ERROR: $*"; exit "$code"; }

allowed() {  # $1 = repo-relative path
  local p
  for p in "${ALLOW[@]}"; do
    case "$1" in "$p"|"$p"/*) return 0 ;; esac
  done
  return 1
}

# ---- message must be plain: no trailers, no attribution
if printf '%s\n' "$MSG" | grep -qiE '^[A-Za-z-]+: .*<.*@.*>|co-authored-by|signed-off-by|generated with|claude|anthropic'; then
  die 4 "commit message looks like it carries a trailer/attribution; refusing: $MSG"
fi

# ---- validate file arguments before touching anything
RELS=()
for f in "$@"; do
  case "$f" in
    /*) abs="$f" ;;
    *) abs="$SRC_ROOT/$f" ;;
  esac
  [ -e "$abs" ] || die 2 "no such file: $f"
  abs="$(cd "$(dirname "$abs")" && pwd)/$(basename "$abs")"
  case "$abs" in
    "$SRC_ROOT"/*) rel="${abs#"$SRC_ROOT"/}" ;;
    *) die 2 "not inside $SRC_ROOT: $f" ;;
  esac
  allowed "$rel" || die 2 "path not publishable (allowed: ${ALLOW[*]}): $rel"
  RELS+=("$rel")
done

# ---- bot clone
git -C "$BOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || die 3 "not a git clone: $BOT"
[ "$(cd "$BOT" && pwd)" != "$SRC_ROOT" ] || die 3 "bot clone must not be the working repo"
br="$(git -C "$BOT" symbolic-ref --short HEAD 2>/dev/null)"
[ "$br" = "main" ] || die 3 "bot clone is on '$br', expected main"
git -C "$BOT" pull --rebase --quiet origin main || die 3 "git pull --rebase failed in $BOT"

# ---- copy
for rel in "${RELS[@]+"${RELS[@]}"}"; do
  mkdir -p "$BOT/$(dirname "$rel")"
  if [ -d "$SRC_ROOT/$rel" ]; then
    mkdir -p "$BOT/$rel"
    cp -Rp "$SRC_ROOT/$rel/." "$BOT/$rel/" || die 3 "copy failed: $rel"
  else
    cp -p "$SRC_ROOT/$rel" "$BOT/$rel" || die 3 "copy failed: $rel"
  fi
  log "copied $rel"
done

# ---- stage only the allowed trees
SPECS=()
for p in "${ALLOW[@]}"; do [ -e "$BOT/$p" ] && SPECS+=("$p"); done
if [ "${#SPECS[@]}" -gt 0 ]; then
  git -C "$BOT" add -A -- "${SPECS[@]}" || die 4 "git add failed"
fi

if git -C "$BOT" diff --cached --quiet; then
  log "nothing new to commit"
else
  git -C "$BOT" -c commit.gpgsign=false -c commit.template= commit --no-verify --quiet \
      --cleanup=strip -m "$MSG" || die 4 "git commit failed"
  body="$(git -C "$BOT" log -1 --format=%B)"
  if printf '%s\n' "$body" | grep -qiE 'co-authored-by|signed-off-by|generated with|claude|anthropic'; then
    git -C "$BOT" reset --soft HEAD~1
    die 4 "a hook added a trailer to the commit message; commit undone"
  fi
  log "committed: $MSG"
fi
SHA="$(git -C "$BOT" rev-parse HEAD)"

# ---- push with retries
pushed=0
for i in 1 2 3; do
  if git -C "$BOT" push --quiet origin HEAD:main; then pushed=1; break; fi
  log "push attempt $i failed; rebasing and retrying"
  sleep $((i * 5))
  git -C "$BOT" pull --rebase --quiet origin main || log "pull --rebase failed on retry $i"
  SHA="$(git -C "$BOT" rev-parse HEAD)"
done
[ "$pushed" = 1 ] || die 5 "push failed after 3 attempts (local HEAD $SHA)"

# ---- verify on origin/main
REMOTE="$(git -C "$BOT" ls-remote origin refs/heads/main | cut -f1)"
if [ "$REMOTE" != "$SHA" ]; then
  # someone else may have pushed on top already; accept if our commit is an ancestor
  git -C "$BOT" fetch --quiet origin main || die 6 "fetch failed while verifying"
  if ! git -C "$BOT" merge-base --is-ancestor "$SHA" "$REMOTE" 2>/dev/null; then
    die 6 "commit $SHA is not on origin/main (ls-remote: ${REMOTE:-none})"
  fi
  log "origin/main is at $REMOTE, which contains $SHA"
fi
log "on origin/main: $SHA"
echo "$SHA"
