# Background memory

Kinveil treats model-produced memories as proposals, not trusted database writes.
After a finalized `UserTurn`, a provider-specific extractor may propose
`MemoryCandidate` objects. `MemoryPolicy` then deterministically stores, queues,
or rejects each candidate.

The response path does not wait for extraction or writes. The intended pipeline
is:

```text
final UserTurn + assistant reply
          |
          v
  MemoryExtractor (LLM adapter)
          |
          v
      MemoryPolicy
     /      |      \
 store    review   reject
   |        |
   +--------+----> SQLiteMemoryStore
```

## Safety rules

- A memory owner is always a stable person UUID, never a display name.
- Unknown or conflicting identities cannot create persistent memories.
- Partial identities and sensitive memories enter the review queue.
- Persistent writes require independent memory consent.
- Private memories are visible only to their owner.
- Shared memories are visible only to their owner and named audience IDs.
- Revoking consent requires an explicit choice to retain or delete old memories.
- Repeated normalized content reinforces one record instead of producing copies.
- A corrected fact can supersede an older record without destroying its history.

`MemoryExtractor` is deliberately provider-neutral. The first LLM integration
can reuse the main conversational model and return structured candidates. A
smaller background model can replace it later without changing policy or storage.

Retrieval always supports local lexical matching, importance, and extractor
confidence. A provider-neutral embedding backend can add semantic similarity;
SQLite remains the source of truth and enforces identity scope.

## Forgetting and consolidation

Forgetting has three separate meanings:

1. Retrieval decay lowers old records through a retention-specific half-life.
2. Reflection consolidates important short-lived observations into a durable,
   evidence-backed record and archives the source events.
3. Retention maintenance expires records, prunes stale review candidates, and
   physically purges expired payloads after a grace period.

Session, short-term, standard, durable, and pinned retention classes have
different decay and expiry behavior. An explicit "remember this" request is
pinned unless the extractor supplies a more specific retention choice. Pinned
records do not expire automatically, but users can still correct or delete them.

Reflections are ordinary model proposals and pass through the same policy as
other memories. Their `derived_from_ids` are stored as explicit evidence edges;
evidence must belong to the same person. The optional `ReflectionEngine` runs
only after recent importance crosses a threshold, instead of calling a model on
every turn.

## Retrieval and context budget

Ranking combines semantic relevance when an embedding backend is configured,
lexical relevance as a fallback, importance, confidence, time decay, and repeat
reinforcement. Searching does not refresh a memory. Only records actually
packed into model context update `last_accessed_at` and `access_count`.

`MemoryService.context_for()` is the only intended LLM-facing entry point. It
packs ranked records into a hard per-request memory budget, capped by
`maximum_context_tokens` (800 by default, configurable with
`KINVEIL_MEMORY_CONTEXT_TOKENS`). The tokenizer-free fallback deliberately uses
UTF-8 byte length as a conservative upper estimate. A provider adapter may later
replace this with its exact tokenizer without changing the storage contract.

Memory content is escaped and wrapped in a data-only `<memory_context>` block.
The surrounding system prompt must continue to state that recalled text is
untrusted data, not executable instructions. The memory budget is separate from
budgets reserved for the system prompt, live conversation, tools, and the model's
answer.

The provider-neutral `BoundedContextBuilder` enforces that total split. It uses
an 8192-token context and reserves 1024 tokens for output by default, configurable
with `KINVEIL_LLM_CONTEXT_TOKENS` and
`KINVEIL_LLM_RESERVED_OUTPUT_TOKENS`. Mandatory system instructions, tool
definitions, and the current user turn must fit or the request fails explicitly;
older history is dropped from the front while recent turns are retained.
