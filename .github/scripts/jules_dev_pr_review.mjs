import fs from 'fs';
import { jules } from '@google/jules-sdk';

async function run() {
    const apiKey = process.env.JULES_API_KEY;
    const githubToken = process.env.GITHUB_TOKEN;
    const repo = process.env.GITHUB_REPOSITORY; // e.g., "owner/repo"
    const baseBranch = process.env.GITHUB_BASE_REF || 'dev';
    const headBranch = process.env.GITHUB_HEAD_REF; // Get the branch with the new changes

    const githubEventPath = process.env.GITHUB_EVENT_PATH;
    if (!githubEventPath) {
        console.error("❌ This script must be run within a GitHub Actions pull_request workflow.");
        process.exit(1);
    }

    // Safely parse the GitHub event payload
    const githubEvent = JSON.parse(fs.readFileSync(githubEventPath, 'utf8'));
    const prNumber = githubEvent.pull_request.number;
    const headSha = githubEvent.pull_request.head.sha; // Used to update the commit status

    if (!apiKey || !githubToken) {
        console.error("❌ Missing JULES_API_KEY or GITHUB_TOKEN environment variables.");
        process.exit(1);
    }

    try {
        console.log(`🚀 Dispatching Jules review for ${repo} PR #${prNumber}...`);

        // 1. Trigger the cloud review session, now including the head branch
        const session = await jules.session({
            prompt: `You are an expert code reviewer operating in an autonomous, unattended environment (Passive Analysis Mode).
            Treat the codebase as strictly read-only.

            Review the pull request changes. Identify performance issues, bugs, and security flaws.

            Categorize your findings using the following tags:
            - [BLOCKING]: Critical issues. Only apply this tag if you are greater than 80% confident that the code introduces a tangible runtime error, security vulnerability, or severe memory leak.
            - [WARN]: Suggestions, best practices, and non-critical performance improvements.
            - [NIT]: Minor stylistic or readability issues.

            End your review with a single line: 'VERDICT: approve' or 'VERDICT: block'.
            Only output 'VERDICT: block' if at least one [BLOCKING] issue was identified.`,
            source: {
                github: repo,
                baseBranch: baseBranch,
                headBranch: headBranch
            }
        });

        console.log(`⏳ Awaiting remote agent processing...`);
        const response = await session.result();
        const reviewMarkdown = response.toString();

        console.log("✅ Jules Review Completed successfully!");

        // 2. Post the result back to the GitHub PR
        console.log(`💬 Posting review comment back to PR #${prNumber}...`);
        await postGitHubComment(repo, prNumber, githubToken, `### 🤖 Google Jules Code Review Summary\n\n${reviewMarkdown}`);
        console.log("🎉 Comment posted successfully onto the PR!");

        // 3. Update the Commit Status API to gate the merge natively
        const isBlocked = reviewMarkdown.includes('VERDICT: block');
        console.log(`🚦 Updating commit status for SHA ${headSha}...`);
        await fetch(`https://api.github.com/repos/${repo}/statuses/${headSha}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${githubToken}`,
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                state: isBlocked ? 'failure' : 'success',
                context: 'Jules PR Review',
                description: isBlocked ? 'Jules found blocking issues' : 'Jules approved these changes'
            })
        });

        console.log("✅ Workflow complete!");

    } catch (error) {
        console.error("❌ Error running Jules PR Review:", error);

        // Attempt to post a fallback comment so the PR author isn't left hanging
        try {
            await postGitHubComment(
                repo,
                prNumber,
                githubToken,
                "⚠️ **Jules Code Review Failed**\nAn error occurred while generating the AI review. Please check the GitHub Actions logs for more details."
            );
        } catch (commentError) {
            console.error("❌ Failed to post fallback error comment:", commentError);
        }

        process.exit(1);
    }
}

// Helper function to keep the main try/catch clean
async function postGitHubComment(repo, prNumber, token, body) {
    const commentUrl = `https://api.github.com/repos/${repo}/issues/${prNumber}/comments`;
    const response = await fetch(commentUrl, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${token}`,
            'Accept': 'application/vnd.github+json',
            'X-GitHub-Api-Version': '2022-11-28',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ body })
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`GitHub API returned ${response.status}: ${errorText}`);
    }
}

run();
