from __future__ import annotations

import pytest
from plane.errors.errors import HttpError


def _not_found() -> HttpError:
    return HttpError("HTTP 404: Not Found", 404, {"error": "Page not found."})


def test_list_falls_back_to_ce_relations_on_dependency_404(registered, spy):
    spy.returns["work_items.dependencies.list"] = _not_found()
    spy.returns["work_items.relations._get"] = {
        "blocking": [{"id": "w-blocking", "relation_type": "blocking"}],
        "blocked_by": ["w-blocked-by"],
        "duplicate": [{"id": "w-duplicate"}],
        "relates_to": [{"id": "w-related"}],
        "start_before": [],
        "start_after": [{"id": "w-start-after", "relation_type": "start_after"}],
        "finish_before": [],
        "finish_after": [],
    }

    result = registered["workitem_relation"].fn(action="list", project_id="p", workitem_id="w")

    assert spy.recorder.methods == ["work_items.dependencies.list", "work_items.relations._get"]
    fallback = spy.recorder.calls[-1]
    assert fallback.kwargs["endpoint"] == "acme/projects/p/work-items/w/relations/"
    assert result["custom"] == {}
    assert result["dependencies"]["blocking"] == [{"id": "w-blocking", "relation_type": "blocking"}]
    assert result["dependencies"]["blocked_by"] == [{"id": "w-blocked-by", "relation_type": "blocked_by"}]
    assert "duplicate" not in result["dependencies"]
    assert "relates_to" not in result["dependencies"]


def test_create_falls_back_to_ce_relations_and_translates_ids(registered, spy):
    spy.returns["work_items.dependencies.create"] = _not_found()
    spy.returns["work_items.relations._post"] = [{"id": "w-2", "relation_type": "blocked_by"}]

    result = registered["workitem_relation"].fn(
        action="create",
        project_id="p",
        workitem_id="w-1",
        workitem_ids=["w-2", "w-3"],
        relation_type="blocking",
    )

    assert spy.recorder.methods == ["work_items.dependencies.create", "work_items.relations._post"]
    fallback = spy.recorder.calls[-1]
    assert fallback.kwargs["endpoint"] == "acme/projects/p/work-items/w-1/relations/"
    assert fallback.kwargs["data"] == {
        "relation_type": "blocking",
        "issues": ["w-2", "w-3"],
    }
    assert result == [{"id": "w-2", "relation_type": "blocked_by"}]


def test_dependency_non_404_is_not_hidden_by_ce_fallback(registered, spy):
    forbidden = HttpError("HTTP 403: Forbidden", 403, {"error": "Permission denied."})
    spy.returns["work_items.dependencies.create"] = forbidden

    with pytest.raises(HttpError) as raised:
        registered["workitem_relation"].fn(
            action="create",
            project_id="p",
            workitem_id="w-1",
            workitem_ids=["w-2"],
            relation_type="blocking",
        )

    assert raised.value is forbidden
    assert spy.recorder.methods == ["work_items.dependencies.create"]


def test_custom_relation_create_stays_on_the_typed_sdk_path(registered, spy):
    registered["workitem_relation"].fn(
        action="create",
        project_id="p",
        workitem_id="w-1",
        workitem_ids=["w-2"],
        relation_definition_id="definition",
        relation_definition_label="Depends on",
    )

    assert spy.recorder.methods == ["work_items.custom_relations.create"]


def test_dependency_delete_does_not_use_the_ce_fallback(registered, spy):
    registered["workitem_relation"].fn(
        action="delete",
        project_id="p",
        workitem_id="w-1",
        related_workitem_id="w-2",
        is_dependency=True,
    )

    assert spy.recorder.methods == ["work_items.dependencies.remove"]
