import {
    loadGitHubContext,
    validateEnv,
    fetchPrDiff,
    buildDiffContext,
    spawnJulesSession,
    postGitHubComment,
    postCommitStatus
} from './jules_utils.mjs';

async function run() {
    const { prNumber, headSha, headRef, prTitle, prBody } = loadGitHubContext();
    const { apiKey, githubToken, repo } = validateEnv();
    const baseBranch = process.env.GITHUB_BASE_REF || 'dev';

    try {
        console.log(`🚀 Fetching PR diff for ${repo} PR #${prNumber}...`);
        const diff = await fetchPrDiff(repo, prNumber, githubToken);

        console.log(`📝 Constructing targeted review prompt...`);
        const diffContext = buildDiffContext(repo, prTitle, prBody, diff);

        const reviewPrompt = `You are an expert code reviewer and release manager. Review the pull request below with high precision and minimal false positives.

# Version Bump Instruction
Before the diff context, assess whether this PR requires a Semantic Versioning bump (patch, minor, or major). 
Check the current \`version\` in \`pyproject.toml\` on the feature branch (\`${headRef}\`) against \`pyproject.toml\` on \`${baseBranch}\`. If it is already bumped, do not perform any file modifications.
If a bump is required and not already done:
1. Canonicalise the version from \`pyproject.toml\`.
2. Update \`version\` in \`pyproject.toml\` to the new SemVer version.
3. Update \`__version__\` in \`pyquantflow/__init__.py\` to match.
4. Run \`uv lock\` to update the lockfile.
5. Create a pull request targeting the \`${headRef}\` branch with title \`chore(release): bump version to X.Y.Z\`.

${diffContext}

# What to review
Focus ONLY on lines changed in this diff. Evaluate for:
- Correctness: logic errors, null/undefined handling, race conditions, edge cases.
- Security: injection risks, hardcoded secrets, auth flaws, sensitive data in logs.
- Reliability: missing error handling, unhandled promise rejections.
- Dependency Version Verification: when evaluating dependency versions, if you do not recognise high version numbers, flag them as a [WARN] stating your cut-off limitation, do not block them as hallucinations.

# Severity tags
Tag each finding EXACTLY one of:
- [BLOCKING] — high-confidence correctness/security flaws (>80% confidence).
- [WARN] — meaningful concerns worth addressing but not blocking.
- [NIT] — small readability or consistency notes (max 3).

# Output format (STRICT)
Respond in Markdown using sections: ## Summary, ## Strengths, ## Findings (grouped by severity heading), and ## SemVer Assessment. End with EXACTLY one line:
\`VERDICT: approve\` — no blocking issues.
\`VERDICT: comment\` — has warnings/nits but nothing blocking.
\`VERDICT: block\` — one or more BLOCKING issues.`;

        console.log(`⏳ Spawning Jules cloud review & autoPr session on branch ${headRef}...`);
        const session = await spawnJulesSession(apiKey, repo, headRef, reviewPrompt, { autoPr: true });

        console.log("✅ Jules session successfully dispatched to cloud environment.");

        console.log(`💬 Posting dispatch comment back to PR #${prNumber}...`);
        await postGitHubComment(
            repo, 
            prNumber, 
            githubToken, 
            `## 🤖 Jules Engaged\n\nA Jules cloud session (\`${session.id}\`) has been dispatched to perform code review and assess SemVer bumping.\n\nIf a version bump is required, Jules will open a separate PR against this feature branch (\`${headRef}\`) with the updated \`pyproject.toml\`, \`__init__.py\`, and \`uv.lock\` files.`
        );

        console.log(`🚦 Updating commit status for SHA ${headSha}...`);
        await postCommitStatus(
            repo,
            headSha,
            githubToken,
            'success',
            'jules/review',
            'Jules review & autoPr session dispatched'
        );

        console.log("✅ Workflow complete!");

    } catch (error) {
        console.error("❌ Error running Jules PR Review:", error);
        try {
            await postGitHubComment(
                repo,
                prNumber,
                githubToken,
                `⚠️ **Jules PR review failed to complete.**\n\nPlease check the GitHub Actions workflow logs for more details.`
            );
        } catch (commentError) {
            console.error("❌ Failed to post fallback error comment:", commentError);
        }
        process.exit(1);
    }
}

run();
