from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kinveil.memory import (
    MemoryCandidate,
    MemoryKind,
    MemoryRecord,
    MemoryService,
    PolicyAction,
    ReflectionEngine,
    RetentionClass,
    Sensitivity,
    SQLiteMemoryStore,
    Visibility,
)


def _candidate(owner_id: str = "person-a", **changes) -> MemoryCandidate:
    values = {
        "content": "用户喜欢喝不加糖的拿铁",
        "owner_id": owner_id,
        "identity_status": "confirmed",
        "confidence": 0.9,
        "importance": 0.7,
    }
    values.update(changes)
    return MemoryCandidate(**values)


def test_memory_requires_consent_and_can_be_approved_later(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    service = MemoryService(store)

    submission = service.submit(_candidate())

    assert submission.decision.action is PolicyAction.REVIEW
    assert submission.review_id is not None
    store.grant_consent("person-a", consent=True)
    memory_id = service.approve_review(
        submission.review_id,
        confirmed_owner_id="person-a",
    )
    recalled = store.recall("person-a", "拿铁")
    assert recalled[0].id == memory_id
    assert recalled[0].content == "用户喜欢喝不加糖的拿铁"


def test_private_and_shared_memories_respect_requester_scope(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    store.add(MemoryRecord("person-a", "我的私人生日计划", 0.8))
    store.add(
        MemoryRecord(
            "person-a",
            "全家周六一起吃晚饭",
            0.8,
            visibility=Visibility.SHARED,
            audience_ids=("person-b",),
        )
    )

    owner_results = store.search("person-a", "计划 晚饭")
    audience_results = store.search("person-b", "计划 晚饭")
    stranger_results = store.search("person-c", "计划 晚饭")

    assert {item.content for item in owner_results} == {
        "我的私人生日计划",
        "全家周六一起吃晚饭",
    }
    assert [item.content for item in audience_results] == ["全家周六一起吃晚饭"]
    assert stranger_results == ()


def test_unknown_identity_is_rejected_and_partial_identity_is_reviewed(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    service = MemoryService(store)

    rejected = service.submit(_candidate(identity_status="unknown"))
    reviewed = service.submit(_candidate(identity_status="partial"))

    assert rejected.decision.action is PolicyAction.REJECT
    assert rejected.review_id is None
    assert reviewed.decision.action is PolicyAction.REVIEW
    assert reviewed.review_id is not None


def test_sensitive_memory_needs_confirmation(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    service = MemoryService(store)

    submission = service.submit(
        _candidate(content="用户正在治疗某种疾病", sensitivity=Sensitivity.SENSITIVE)
    )

    assert submission.decision.action is PolicyAction.REVIEW
    service.approve_review(submission.review_id, confirmed_owner_id="person-a")
    assert store.recall("person-a", "治疗")[0].sensitivity is Sensitivity.SENSITIVE


def test_duplicate_memory_reinforces_one_record(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)

    first = store.add(MemoryRecord("person-a", "喜欢蓝色", 0.4, confidence=0.7))
    second = store.add(MemoryRecord("person-a", " 喜欢蓝色 ", 0.9, confidence=0.95))

    assert first == second
    results = store.recall("person-a", "蓝色")
    assert len(results) == 1
    assert results[0].importance == 0.9
    assert results[0].confidence == 0.95


def test_superseding_memory_hides_old_value(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    old_id = store.add(MemoryRecord("person-a", "用户住在上海", 0.8))

    store.add(
        MemoryRecord(
            "person-a",
            "用户现在住在杭州",
            0.9,
            supersedes_id=old_id,
        )
    )

    assert [item.content for item in store.recall("person-a", "住在")] == [
        "用户现在住在杭州"
    ]


def test_revoke_consent_requires_explicit_retention_choice(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    store.add(MemoryRecord("person-a", "保留的记忆", 0.7))

    assert store.revoke_consent("person-a", delete_memories=False)
    assert store.recall("person-a", "保留")[0].content == "保留的记忆"

    store.grant_consent("person-a", consent=True)
    assert store.revoke_consent("person-a", delete_memories=True)
    assert store.recall("person-a", "保留") == ()


def test_context_has_a_hard_token_budget_and_marks_only_injected_memories(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    for index in range(6):
        store.add(MemoryRecord("person-a", f"用户的第{index}条长期偏好记录", 0.8))
    service = MemoryService(store, maximum_context_tokens=120)

    context = service.context_for("person-a", "长期偏好", max_tokens=1000)

    assert context.estimated_tokens <= 120
    assert context.max_tokens == 120
    assert context.truncated
    assert 0 < len(context.records) < 6
    records = store.recall("person-a", "长期偏好", limit=10)
    accessed = {item.id for item in records if item.access_count == 1}
    assert accessed == {item.id for item in context.records}


def test_recent_memory_ranks_above_stale_equivalent_memory(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    now = datetime.now(UTC)
    store.add(
        MemoryRecord(
            "person-a",
            "项目状态旧记录",
            0.6,
            last_observed_at=now - timedelta(days=180),
            retention=RetentionClass.STANDARD,
        )
    )
    store.add(
        MemoryRecord(
            "person-a",
            "项目状态新记录",
            0.6,
            last_observed_at=now,
            retention=RetentionClass.STANDARD,
        )
    )

    assert store.recall("person-a", "项目状态")[0].content == "项目状态新记录"


def test_explicit_memory_is_pinned_and_conflict_supersedes_old_fact(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    service = MemoryService(store)
    first = service.submit(_candidate(content="用户住在上海", topic_key="home_city"))

    second = service.submit(
        _candidate(
            content="用户现在住在杭州",
            topic_key="home_city",
            requested_by_user=True,
        )
    )

    assert first.memory_id is not None
    assert second.memory_id is not None
    active = store.recall("person-a", "住在")
    assert [item.content for item in active] == ["用户现在住在杭州"]
    assert active[0].retention is RetentionClass.PINNED
    assert active[0].expires_at is None


def test_maintenance_expires_then_purges_and_closes_session_memory(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    now = datetime.now(UTC)
    store.add(
        MemoryRecord(
            "person-a",
            "已经过期的事件",
            0.5,
            expires_at=now - timedelta(seconds=1),
        )
    )
    store.add(
        MemoryRecord(
            "person-a",
            "只在当前会话保留",
            0.5,
            retention=RetentionClass.SESSION,
        )
    )

    assert store.maintain(now=now)["expired"] == 1
    assert store.close_session_memories("person-a") == 1
    later = now + timedelta(days=31)
    assert store.maintain(now=later)["purged"] == 2


class _KeywordEmbedding:
    model_id = "test-keywords-v1"

    def embed(self, text: str):
        if "咖啡" in text or "饮品" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


def test_optional_embedding_backend_provides_semantic_relevance(tmp_path) -> None:
    store = SQLiteMemoryStore(
        tmp_path / "memory.db", embedding_backend=_KeywordEmbedding()
    )
    store.grant_consent("person-a", consent=True)
    store.add(MemoryRecord("person-a", "用户每天喝咖啡", 0.5))
    store.add(MemoryRecord("person-a", "用户喜欢踢足球", 0.5))

    assert store.recall("person-a", "常喝的饮品")[0].content == "用户每天喝咖啡"


class _Reflector:
    def reflect(self, owner_id, evidence):
        return (
            _candidate(
                owner_id,
                content="用户持续关注低延迟的实时交互体验",
                kind=MemoryKind.REFLECTION,
                confidence=0.85,
                importance=0.9,
                retention=RetentionClass.DURABLE,
            ),
        )


def test_reflection_is_triggered_by_importance_and_keeps_evidence(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    store.grant_consent("person-a", consent=True)
    store.add(
        MemoryRecord(
            "person-a",
            "用户要求降低语音延迟",
            0.8,
            kind=MemoryKind.EVENT,
            retention=RetentionClass.SHORT_TERM,
        )
    )
    store.add(
        MemoryRecord(
            "person-a",
            "用户不希望对话有顿挫",
            0.8,
            kind=MemoryKind.EVENT,
            retention=RetentionClass.SHORT_TERM,
        )
    )
    engine = ReflectionEngine(
        MemoryService(store),
        _Reflector(),
        importance_threshold=1.5,
    )

    submissions = engine.run_if_due("person-a")

    assert submissions[0].decision.action is PolicyAction.STORE
    reflection = store.recall("person-a", "实时交互")[0]
    assert reflection.kind is MemoryKind.REFLECTION
    assert len(reflection.derived_from_ids) == 2
    assert all(item.kind is not MemoryKind.EVENT for item in store.recall("person-a", "延迟"))
    assert engine.run_if_due("person-a") == ()
