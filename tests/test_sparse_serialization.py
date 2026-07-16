import asyncio
from types import SimpleNamespace

import pytest
from fastmcp import Client, FastMCP
from plane.models.intake import IntakeWorkItem
from plane.models.work_items import WorkItem, WorkItemDetail, WorkItemSearch, WorkItemSearchItem

from plane_mcp.tools.cycles import register_cycle_tools
from plane_mcp.tools.intake import register_intake_tools
from plane_mcp.tools.modules import register_module_tools
from plane_mcp.tools.serialization import parse_requested_fields, serialize_resource, serialize_resources
from plane_mcp.tools.work_items import register_work_item_tools


def _tool_fn(register, name):
    async def get_fn():
        mcp = FastMCP("test")
        register(mcp)
        tool = await mcp.get_tool(name)
        return tool.fn

    return asyncio.run(get_fn())


async def _call_tool_with_sync_shim(register, tool_name, arguments):
    mcp = FastMCP("test")
    register(mcp)
    tool = await mcp.get_tool(tool_name)
    original = tool.fn

    # FastMCP's in-process client times out on sync tools in this Python 3.14 test runtime.
    async def async_fn(*args, **kwargs):
        return original(*args, **kwargs)

    tool.fn = async_fn
    async with Client(mcp, timeout=5) as client:
        return await client.call_tool(tool_name, arguments)


def _paginated_response(results):
    return SimpleNamespace(
        results=results,
        total_count=10,
        count=len(results),
        next_cursor="next",
        prev_cursor="prev",
        next_page_results=True,
        prev_page_results=False,
        total_pages=5,
        total_results=10,
    )


def _work_item(item_id="wi-1"):
    return WorkItem.model_validate({"id": item_id, "name": "One", "target_date": None, "priority": None})


def _work_item_detail(item_id="wi-1"):
    return WorkItemDetail.model_validate({"id": item_id, "name": "One", "target_date": None, "priority": None})


def test_parse_requested_fields_trims_and_ignores_empty_entries():
    assert parse_requested_fields("id, name") == {"id", "name"}
    assert parse_requested_fields("id,,name,") == {"id", "name"}


def test_empty_non_null_fields_string_is_rejected():
    with pytest.raises(ValueError, match="at least one field"):
        parse_requested_fields(" , ,, ")


def test_serialize_resource_projects_requested_fields_and_preserves_requested_null():
    assert serialize_resource(_work_item(), fields="id,name,target_date") == {
        "id": "wi-1",
        "name": "One",
        "target_date": None,
    }


def test_serialize_resource_omits_unrequested_null_fields():
    assert serialize_resource(_work_item(), fields="id,name") == {"id": "wi-1", "name": "One"}


def test_serialize_resource_without_fields_preserves_full_shape():
    data = serialize_resource(_work_item())

    assert data["id"] == "wi-1"
    assert data["name"] == "One"
    assert "target_date" in data
    assert data["target_date"] is None


def test_serialize_mapping_and_sequence_resources():
    resource = {"id": "wi-1", "name": "One", "target_date": None, "priority": None}

    assert serialize_resource(resource, fields="id,target_date") == {
        "id": "wi-1",
        "target_date": None,
    }
    assert serialize_resources([resource], fields="id") == [{"id": "wi-1"}]


def test_list_work_items_projects_each_result_and_preserves_pagination(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "list_work_items")
    response = _paginated_response([_work_item("wi-1"), _work_item("wi-2")])
    seen = {}

    def list_work_items(**kwargs):
        seen["fields"] = kwargs["params"].fields
        return response

    work_items = SimpleNamespace(list=list_work_items, list_workspace=lambda **kwargs: response)
    client = SimpleNamespace(work_items=work_items)
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(project_id="project", fields="id,target_date")

    assert result["results"] == [{"id": "wi-1", "target_date": None}, {"id": "wi-2", "target_date": None}]
    assert result["total_count"] == 10
    assert result["count"] == 2
    assert result["next_cursor"] == "next"
    assert result["prev_cursor"] == "prev"
    assert result["next_page_results"] is True
    assert result["prev_page_results"] is False
    assert seen["fields"] == "id,target_date"


def test_list_work_items_without_fields_preserves_full_result_shape(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "list_work_items")
    response = _paginated_response([_work_item()])
    work_items = SimpleNamespace(list=lambda **kwargs: response, list_workspace=lambda **kwargs: response)
    client = SimpleNamespace(work_items=work_items)
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(project_id="project")

    assert result["results"][0]["id"] == "wi-1"
    assert "target_date" in result["results"][0]
    assert result["results"][0]["target_date"] is None


def test_retrieve_work_item_returns_full_and_sparse_dicts(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "retrieve_work_item")
    work_item = _work_item_detail()
    seen = {}

    def retrieve(**kwargs):
        seen["fields"] = kwargs["params"].fields
        return work_item

    client = SimpleNamespace(work_items=SimpleNamespace(retrieve=retrieve))
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    full = fn(project_id="project", work_item_id="wi-1")
    result = fn(project_id="project", work_item_id="wi-1", fields="id,target_date")

    assert full == work_item.model_dump()
    assert "priority" in full
    assert result == {"id": "wi-1", "target_date": None}
    assert seen["fields"] == "id,target_date"


def test_retrieve_work_item_by_identifier_returns_sparse_dict(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "retrieve_work_item_by_identifier")
    work_item = _work_item_detail()
    seen = {}

    def retrieve_by_identifier(**kwargs):
        seen["issue_identifier"] = kwargs["issue_identifier"]
        seen["fields"] = kwargs["params"].fields
        return work_item

    client = SimpleNamespace(work_items=SimpleNamespace(retrieve_by_identifier=retrieve_by_identifier))
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(work_item_identifier="PROJ-7", fields="id,target_date")

    assert result == {"id": "wi-1", "target_date": None}
    assert seen == {"issue_identifier": 7, "fields": "id,target_date"}


def test_list_archived_work_items_projects_results(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "list_archived_work_items")
    response = _paginated_response([_work_item()])
    seen = {}

    def list_archived(**kwargs):
        seen["fields"] = kwargs["params"].fields
        return response

    client = SimpleNamespace(work_items=SimpleNamespace(list_archived=list_archived))
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(project_id="project", fields="id,target_date")

    assert result["results"] == [{"id": "wi-1", "target_date": None}]
    assert result["total_count"] == 10
    assert seen["fields"] == "id,target_date"


def test_search_work_items_projects_each_issue_and_preserves_no_fields_model(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "search_work_items")
    search_item = WorkItemSearchItem(
        id="wi-1",
        name="One",
        sequence_id="1",
        project__identifier="PROJ",
        project_id="project",
        workspace__slug="workspace",
        target_date=None,
    )
    response = WorkItemSearch(issues=[search_item])
    seen = {}

    def search(**kwargs):
        seen["fields"] = kwargs["params"].fields
        return response

    client = SimpleNamespace(work_items=SimpleNamespace(search=search))
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    assert fn(query="one") == response.model_dump()
    result = fn(query="one", fields="id,target_date")

    assert result["issues"] == [{"id": "wi-1", "target_date": None}]
    assert seen["fields"] == "id,target_date"


@pytest.mark.parametrize(
    ("register", "tool_name", "client_attr", "method_name", "extra_kwargs", "patch_target"),
    [
        (
            register_cycle_tools,
            "list_cycle_work_items",
            "cycles",
            "list_work_items",
            {"cycle_id": "cycle"},
            "plane_mcp.tools.cycles.get_plane_client_context",
        ),
        (
            register_module_tools,
            "list_module_work_items",
            "modules",
            "list_work_items",
            {"module_id": "module"},
            "plane_mcp.tools.modules.get_plane_client_context",
        ),
    ],
)
def test_cycle_and_module_work_item_lists_project_each_result(
    monkeypatch, register, tool_name, client_attr, method_name, extra_kwargs, patch_target
):
    fn = _tool_fn(register, tool_name)
    response = _paginated_response([_work_item()])
    seen = {}

    def list_items(**kwargs):
        seen["fields"] = kwargs["params"].fields
        return response

    resource = SimpleNamespace(**{method_name: list_items})
    client = SimpleNamespace(**{client_attr: resource})
    monkeypatch.setattr(patch_target, lambda: (client, "workspace"))

    result = fn(project_id="project", fields="id,target_date", **extra_kwargs)

    assert result["results"] == [{"id": "wi-1", "target_date": None}]
    assert result["total_count"] == 10
    assert seen["fields"] == "id,target_date"


def test_intake_list_and_retrieve_use_fields_from_existing_params(monkeypatch):
    list_fn = _tool_fn(register_intake_tools, "list_intake_work_items")
    retrieve_fn = _tool_fn(register_intake_tools, "retrieve_intake_work_item")
    intake_item = IntakeWorkItem.model_validate({"id": "intake-1", "issue": "wi-1", "source_email": None})
    seen = {}

    def list_intake(**kwargs):
        if kwargs["params"] is not None:
            seen["list_fields"] = kwargs["params"].fields
        return SimpleNamespace(results=[intake_item])

    def retrieve_intake(**kwargs):
        if kwargs["params"] is not None:
            seen["retrieve_fields"] = kwargs["params"].fields
        return intake_item

    client = SimpleNamespace(intake=SimpleNamespace(list=list_intake, retrieve=retrieve_intake))
    monkeypatch.setattr("plane_mcp.tools.intake.get_plane_client_context", lambda: (client, "workspace"))

    assert list_fn(project_id="project") == [intake_item.model_dump()]
    assert retrieve_fn(project_id="project", work_item_id="wi-1") == intake_item.model_dump()
    assert list_fn(project_id="project", params={"fields": "id,source_email"}) == [
        {"id": "intake-1", "source_email": None}
    ]
    assert retrieve_fn(project_id="project", work_item_id="wi-1", params={"fields": "id,source_email"}) == {
        "id": "intake-1",
        "source_email": None,
    }
    assert seen == {"list_fields": "id,source_email", "retrieve_fields": "id,source_email"}


def test_tool_schemas_reflect_sparse_dict_outputs():
    async def schema_fields():
        mcp = FastMCP("test")
        register_work_item_tools(mcp)
        retrieve = await mcp.get_tool("retrieve_work_item")
        search = await mcp.get_tool("search_work_items")
        return retrieve.parameters["properties"], retrieve.output_schema, search.output_schema

    properties, retrieve_output, search_output = asyncio.run(schema_fields())

    assert "fields" in properties
    assert retrieve_output["additionalProperties"] is True
    assert search_output["additionalProperties"] is True


def test_fastmcp_client_full_retrieve_work_item_matches_upstream_shape(monkeypatch):
    async def call_tool():
        work_items = SimpleNamespace(retrieve=lambda **kwargs: _work_item_detail())
        client_context = SimpleNamespace(work_items=work_items)
        monkeypatch.setattr(
            "plane_mcp.tools.work_items.get_plane_client_context",
            lambda: (client_context, "workspace"),
        )
        return await _call_tool_with_sync_shim(
            register_work_item_tools,
            "retrieve_work_item",
            {"project_id": "project", "work_item_id": "wi-1"},
        )

    result = asyncio.run(call_tool())

    assert result.structured_content == _work_item_detail().model_dump()
    assert "result" not in result.structured_content
    assert "priority" in result.structured_content


def test_fastmcp_client_sparse_retrieve_work_item_has_no_result_wrapper(monkeypatch):
    async def call_tool():
        work_items = SimpleNamespace(retrieve=lambda **kwargs: _work_item_detail())
        client_context = SimpleNamespace(work_items=work_items)
        monkeypatch.setattr(
            "plane_mcp.tools.work_items.get_plane_client_context",
            lambda: (client_context, "workspace"),
        )
        return await _call_tool_with_sync_shim(
            register_work_item_tools,
            "retrieve_work_item",
            {"project_id": "project", "work_item_id": "wi-1", "fields": "id,target_date"},
        )

    result = asyncio.run(call_tool())

    assert result.structured_content == {"id": "wi-1", "target_date": None}
    assert "result" not in result.structured_content
    assert "priority" not in result.structured_content


def test_fastmcp_client_full_search_work_items_matches_upstream_shape(monkeypatch):
    search_item = WorkItemSearchItem(
        id="wi-1",
        name="One",
        sequence_id="1",
        project__identifier="PROJ",
        project_id="project",
        workspace__slug="workspace",
        target_date=None,
    )
    response = WorkItemSearch(issues=[search_item])

    async def call_tool():
        client_context = SimpleNamespace(work_items=SimpleNamespace(search=lambda **kwargs: response))
        monkeypatch.setattr(
            "plane_mcp.tools.work_items.get_plane_client_context",
            lambda: (client_context, "workspace"),
        )
        return await _call_tool_with_sync_shim(register_work_item_tools, "search_work_items", {"query": "one"})

    result = asyncio.run(call_tool())

    assert result.structured_content == response.model_dump()
    assert "result" not in result.structured_content


def test_fastmcp_client_sparse_search_work_items_has_no_result_wrapper(monkeypatch):
    search_item = WorkItemSearchItem(
        id="wi-1",
        name="One",
        sequence_id="1",
        project__identifier="PROJ",
        project_id="project",
        workspace__slug="workspace",
        target_date=None,
    )
    response = WorkItemSearch(issues=[search_item])

    async def call_tool():
        client_context = SimpleNamespace(work_items=SimpleNamespace(search=lambda **kwargs: response))
        monkeypatch.setattr(
            "plane_mcp.tools.work_items.get_plane_client_context",
            lambda: (client_context, "workspace"),
        )
        return await _call_tool_with_sync_shim(
            register_work_item_tools,
            "search_work_items",
            {"query": "one", "fields": "id,target_date"},
        )

    result = asyncio.run(call_tool())

    assert result.structured_content == {"issues": [{"id": "wi-1", "target_date": None}]}
    assert "result" not in result.structured_content


def test_fastmcp_client_full_retrieve_intake_matches_upstream_shape(monkeypatch):
    async def call_tool():
        intake_item = IntakeWorkItem.model_validate({"id": "intake-1", "issue": "wi-1", "source_email": None})
        client_context = SimpleNamespace(intake=SimpleNamespace(retrieve=lambda **kwargs: intake_item))
        monkeypatch.setattr(
            "plane_mcp.tools.intake.get_plane_client_context",
            lambda: (client_context, "workspace"),
        )
        return await _call_tool_with_sync_shim(
            register_intake_tools,
            "retrieve_intake_work_item",
            {"project_id": "project", "work_item_id": "wi-1"},
        )

    result = asyncio.run(call_tool())
    expected = IntakeWorkItem.model_validate({"id": "intake-1", "issue": "wi-1", "source_email": None}).model_dump()

    assert result.structured_content == expected
    assert "result" not in result.structured_content
    assert "issue" in result.structured_content


def test_fastmcp_client_sparse_retrieve_intake_has_no_result_wrapper(monkeypatch):
    async def call_tool():
        intake_item = IntakeWorkItem.model_validate({"id": "intake-1", "issue": "wi-1", "source_email": None})
        client_context = SimpleNamespace(intake=SimpleNamespace(retrieve=lambda **kwargs: intake_item))
        monkeypatch.setattr(
            "plane_mcp.tools.intake.get_plane_client_context",
            lambda: (client_context, "workspace"),
        )
        return await _call_tool_with_sync_shim(
            register_intake_tools,
            "retrieve_intake_work_item",
            {"project_id": "project", "work_item_id": "wi-1", "params": {"fields": "id,source_email"}},
        )

    result = asyncio.run(call_tool())

    assert result.structured_content == {"id": "intake-1", "source_email": None}
    assert "result" not in result.structured_content
    assert "issue" not in result.structured_content


def test_fastmcp_client_full_intake_list_keeps_upstream_result_wrapper(monkeypatch):
    intake_item = IntakeWorkItem.model_validate({"id": "intake-1", "issue": "wi-1", "source_email": None})

    async def call_tool():
        client_context = SimpleNamespace(
            intake=SimpleNamespace(list=lambda **kwargs: SimpleNamespace(results=[intake_item]))
        )
        monkeypatch.setattr(
            "plane_mcp.tools.intake.get_plane_client_context",
            lambda: (client_context, "workspace"),
        )
        return await _call_tool_with_sync_shim(
            register_intake_tools,
            "list_intake_work_items",
            {"project_id": "project"},
        )

    result = asyncio.run(call_tool())

    assert result.structured_content == {"result": [intake_item.model_dump()]}
