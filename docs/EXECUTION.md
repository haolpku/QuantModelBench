# Tools and execution boundaries

`shared/agentic/tools.py` exposes `python_exec`, `read_artifact`, and
`write_artifact`. Exploration runs with a public payload and case workspace.
Final candidate execution is handled by the Task C code runner, also reused by
Task B. Source policy checks, output validation, timeouts and audit records are
research guardrails, not a security proof.

`python -I` isolates Python import/environment behavior; it does not by itself
deny filesystem access or network access. A copied public workspace and AST
restrictions must not be described as a hardened sandbox.

For execution of arbitrary external submissions, deploy an additional container
or VM boundary with:

- only the allowed public or per-origin inputs mounted read-only;
- no evaluator truth, provider credentials, original identifier mappings or full
  future panels accessible to generated code;
- network disabled and CPU, memory, process, disk and wall-time limits;
- separate writable scratch space for each run;
- score computation in a trusted evaluator outside the generated-code worker.

The same restrictions are needed during rolling: ending the LLM conversation does
not make its submitted Python trustworthy. Pass each origin's permitted data to a
worker and score its outputs externally. Do not mount the entire HF download into
an agent container. A provider token is needed by the orchestrator, not by the
generated-code process.

This release does not add or claim a hardened container implementation. Harbor
compatibility is a possible future adapter; the current callable tools and
frozen-program protocol remain unchanged.
