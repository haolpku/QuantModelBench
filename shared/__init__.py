"""Repository-level shared code used by more than one task package.

Task B and Task C each ship their own ``factoropt`` package (the names must
stay identical so every task directory remains self-contained).  Anything that
is genuinely task-neutral therefore cannot live inside either one of them, or
the other task has to reach across the filesystem to load it.  This package is
that neutral home.

``shared.agentic`` holds the provider-neutral Chat Completions tool loop and
its bounded tools: no task names, validators or business rules appear there.
"""
