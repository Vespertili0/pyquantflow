import {
    loadGitHubContext,
    validateEnv,
    fetchPrDiff,
    buildDiffContext,
    spawnJulesSession,
    awaitSessionResult,
    postGitHubComment,
    postCommitStatus
} from './jules_utils.mjs';

async function run() {
    const { prNumber, headSha, prTitle, prBody } = loadGitHubContext();
    const { apiKey, githubToken, repo } = validateEnv();
    const baseBranch = process.env.GITHUB_BASE_REF || 'dev';

    try {
        console.log(`🚀 Fetching PR diff for ${repo} PR #${prNumber}...`);
        const diff = await fetchPrDiff(repo, prNumber, githubToken);

        console.log(`📝 Constructing targeted review prompt...`);
        const diffContext = buildDiffContext(repo, prTitle, prBody, diff);

        const reviewPrompt = `You are an expert code reviewer. Review the pull request below with high precision and minimal false positives.

${diffContext}

# What to review
Focus ONLY on lines changed in this diff. Evaluate for:
- Correctness: logic errors, null/undefined handling, race conditions, edge cases.
- Security: injection risks, hardcoded secrets, auth flaws, sensitive data in logs.
- Reliability: missing error handling, unhandled promise rejections.
- Dependency Version Verification: when evaluating dependency versions (e.g., in pyproject.toml, package.json, or requirements.txt), be aware that your internal knowledge has a cut-off date. Do not confidently flag unrecognised high version numbers as [BLOCKING] hallucinations or supply chain attacks solely because you do not recognise them. Instead, flag them as a [WARN], explicitly state that your assessment is limited by your training data cut-off, and recommend that the author or CI/CD pipelines verify the availability of these versions.

# Severity tags
Tag each finding EXACTLY one of:
- [BLOCKING] — high-confidence correctness/security flaws (>80% confidence).
- [WARN] — meaningful concerns worth addressing but not blocking.
- [NIT] — small readability or consistency notes (max 3).

# Output format (STRICT)
Respond in Markdown using sections: ## Summary, ## Strengths, ## Findings (grouped by severity heading), and end with EXACTLY one line:
\`VERDICT: approve\` — no blocking issues.
\`VERDICT: comment\` — has warnings/nits but nothing blocking.
\`VERDICT: block\` — one or more BLOCKING issues.`;

        console.log(`⏳ Spawning Jules cloud review session...`);
        const session = await spawnJulesSession(apiKey, repo, baseBranch, reviewPrompt);

        const reviewMarkdown = await awaitSessionResult(session);

        console.log("✅ Jules Review Completed successfully!");

        console.log(`💬 Posting review comment back to PR #${prNumber}...`);
        await postGitHubComment(repo, prNumber, githubToken, `## 🤖 Jules Review\n\n${reviewMarkdown}\n\n---\n_Session: \`${session.id}\`_`);

        const isBlocked = reviewMarkdown.toUpperCase().includes('VERDICT: BLOCK') || reviewMarkdown.includes('[BLOCKING]');
        console.log(`🚦 Updating commit status for SHA ${headSha}...`);
        await postCommitStatus(
            repo,
            headSha,
            githubToken,
            isBlocked ? 'failure' : 'success',
            'jules/review',
            isBlocked ? 'Blocking issues found by Jules' : 'Review complete (verdict: approve)'
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
