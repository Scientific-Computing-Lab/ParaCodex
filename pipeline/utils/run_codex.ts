import { Codex } from "@openai/codex-sdk";
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

// ES module equivalents for __dirname and __filename
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Parse arguments
// Usage: node run_codex.ts <prompt_file> <workdir>
const promptFile = process.argv[2];
const workdir = process.argv[3];

if (!promptFile || !workdir) {
    console.error("Usage: node run_codex.ts <prompt_file> <workdir>");
    process.exit(1);
}

async function main() {
    try {
        let prompt = fs.readFileSync(promptFile, 'utf-8');

        // Use codex-hpc from PATH (or override via CODEX_HPC_PATH) to avoid relative paths.
        // This makes the pipeline usable regardless of where `vendor/` lives.
        const customBinaryPath = (() => {
            if (process.env.CODEX_HPC_PATH && process.env.CODEX_HPC_PATH.trim().length > 0) {
                return process.env.CODEX_HPC_PATH.trim();
            }
            try {
                return execSync('command -v codex', { stdio: ['ignore', 'pipe', 'ignore'] })
                    .toString()
                    .trim();
            } catch {
                // Fall back to relying on PATH resolution by the SDK spawn (best-effort).
                return 'codex';
            }
        })();

        // Pass environment variables to the Codex binary
        const codex = new Codex({
            codexPathOverride: customBinaryPath,
            env: process.env as Record<string, string>
        });

        const thread = codex.startThread({
            model: process.env.CODEX_MODEL,
            sandboxMode: 'danger-full-access',
            workingDirectory: path.resolve(workdir),
            skipGitRepoCheck: true,
            // Additional options from CLI defaults
            approvalPolicy: 'never', // Equivalent to -a never (which exec implies?)
        });

        const result = await thread.run(prompt);

        // CUSTOM HPC: Copy rollout trace file if requested
        const traceOutput = process.env.CODEX_TRACE_OUTPUT;
        const threadId = thread.id;
        if (traceOutput && threadId) {
            try {
                // Rollout files are saved to ~/.codex/sessions/YYYY/MM/DD/rollout-*-{threadId}.jsonl
                const homeDir = process.env.HOME || process.env.USERPROFILE || os.homedir();
                const today = new Date();
                const year = today.getFullYear();
                const month = String(today.getMonth() + 1).padStart(2, '0');
                const day = String(today.getDate()).padStart(2, '0');
                const sessionsDir = path.join(homeDir, '.codex', 'sessions', String(year), month, day);

                // Find rollout file matching this thread ID
                if (fs.existsSync(sessionsDir)) {
                    const files = fs.readdirSync(sessionsDir);
                    const rolloutFile = files.find(f => f.startsWith('rollout-') && f.includes(threadId) && f.endsWith('.jsonl'));

                    if (rolloutFile) {
                        const sourcePath = path.join(sessionsDir, rolloutFile);

                        // Make filename unique by appending thread ID to avoid overwrites
                        let destPath: string;
                        const tracePath = path.resolve(traceOutput);

                        // Check if path exists and is a directory
                        let isDirectory = false;
                        try {
                            isDirectory = fs.statSync(tracePath).isDirectory();
                        } catch {
                            // Path doesn't exist - check if it looks like a directory (no extension)
                            isDirectory = !path.extname(tracePath);
                        }

                        if (isDirectory) {
                            // If it's a directory, create file with thread ID inside
                            destPath = path.join(tracePath, `codex-trace-${threadId}.jsonl`);
                        } else {
                            // If it's a file path, insert thread ID before extension
                            const ext = path.extname(tracePath);
                            const base = tracePath.slice(0, -ext.length);
                            destPath = `${base}-${threadId}${ext}`;
                        }

                        fs.copyFileSync(sourcePath, destPath);

                        // Also write prompt to file with same naming pattern
                        const promptPath = destPath.replace(/\.jsonl$/, '-prompt.txt');
                        fs.writeFileSync(promptPath, prompt, 'utf-8');
                    }
                }
            } catch (err: any) {
                // Non-fatal: log but don't fail the run
                console.error(JSON.stringify({
                    warning: `Failed to copy trace file: ${err.message}`,
                }));
            }
        }

        // serialize result to stdout
        console.log(JSON.stringify(result, null, 2));

    } catch (error: any) {
        console.error(JSON.stringify({
            error: error.message || String(error),
            stack: error.stack
        }));
        process.exit(1);
    }
}

main();
