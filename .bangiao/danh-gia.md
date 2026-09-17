The data and evaluation changes appear to satisfy the stated metrics, but the review wrapper now has a persistent Git index side effect that can disrupt the handoff workflow. This should be fixed before considering the patch correct.

Review comment:

- [P2] Avoid mutating the Git index during review — /Users/nguyenvokhang/Downloads/vsf-vietnamese-rag-evaluation/scripts/codex_review.sh:12-12
  When this wrapper is run from a tree with untracked files, `git add -A -N` leaves intent-to-add entries in the user's index even though the script is advertised as review-only/read-only. That changes `git status` and can affect subsequent diffs or commits after the review exits or fails; it also expands the change scope beyond the handoff plan's allowed files (`.bangiao/ke-hoach.md:47-53`). Use a non-mutating way to surface untracked diffs or clean up the index afterward.