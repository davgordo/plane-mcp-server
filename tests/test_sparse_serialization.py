import asyncio
from types import SimpleNamespace

import pytest
from fastmcp import FastMCP
from plane.models.intake import IntakeWorkItem
from plane.models.projects import Project
from plane.models.work_items import WorkItem, WorkItemDetail, WorkItemSearch, WorkItemSearchItem

from plane_mcp.tools.cycles import register_cycle_tools
from plane_mcp.tools.intake import register_intake_tools
from plane_mcp.tools.modules import register_module_tools
from plane_mcp.tools.projects import register_project_tools
from plane_mcp.tools.serialization import parse_requested_fields, serialize_resource, serialize_resources
from plane_mcp.tools.work_items import register_work_item_tools


def _tool_fn(register, name):
    async def get_fn():
        mcp = FastMCP("test")
        register(mcp)
        tool = await mcp.get_tool(name)
        return tool.fn

    return asyncio.run(get_fn())


def _paginated_response(results):
    return SimpleNamespace(
        results=results,
        total_count=10,
        count=len(results),
        next_cursor="next",
        prev_cursor="prev",
        next_page_results=True,
        prev_page_results=False,
    )


def test_parse_requested_fields_trims_and_ignores_empty_entries():
    assert parse_requested_fields("id, name") == {"id", "name"}
    assert parse_requested_fields("id,,name,") == {"id", "name"}


def test_empty_non_null_fields_string_is_rejected():
    with pytest.raises(ValueError, match="at least one field"):
        parse_requested_fields(" , ,, ")


def test_serialize_resource_projects_requested_fields_and_preserves_requested_null():
    item = WorkItem.model_validate({"id": "wi-1", "name": "Item", "target_date": None})

    assert serialize_resource(item, fields="id,name,target_date") == {
        "id": "wi-1",
        "name": "Item",
        "target_date": None,
    }


def test_serialize_resource_omits_unrequested_null_fields():
    item = WorkItem.model_validate({"id": "wi-1", "name": "Item", "target_date": None})

    assert serialize_resource(item, fields="id,name") == {"id": "wi-1", "name": "Item"}


def test_serialize_resource_without_fields_preserves_full_shape():
    item = WorkItem.model_validate({"id": "wi-1", "name": "Item"})
    data = serialize_resource(item)

    assert data["id"] == "wi-1"
    assert data["name"] == "Item"
    assert "target_date" in data
    assert data["target_date"] is None


def test_serialize_mapping_resources_without_failing():
    resource = {"id": "wi-1", "name": "Item", "target_date": None, "priority": None}

    assert serialize_resource(resource, fields="id,target_date") == {
        "id": "wi-1",
        "target_date": None,
    }
    assert serialize_resources([resource], fields="id") == [{"id": "wi-1"}]


def test_list_work_items_projects_each_result_and_preserves_pagination(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "list_work_items")
    response = _paginated_response(
        [
            WorkItem.model_validate({"id": "wi-1", "name": "One", "target_date": None}),
            WorkItem.model_validate({"id": "wi-2", "name": "Two", "target_date": None}),
        ]
    )
    work_items = SimpleNamespace(list=lambda **kwargs: response, list_workspace=lambda **kwargs: response)
    client = SimpleNamespace(work_items=work_items)
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(project_id="project", fields="id,name")

    assert result["results"] == [{"id": "wi-1", "name": "One"}, {"id": "wi-2", "name": "Two"}]
    assert result["total_count"] == 10
    assert result["count"] == 2
    assert result["next_cursor"] == "next"
    assert result["prev_cursor"] == "prev"
    assert result["next_page_results"] is True
    assert result["prev_page_results"] is False


def test_list_cycle_work_items_projects_each_result(monkeypatch):
    fn = _tool_fn(register_cycle_tools, "list_cycle_work_items")
    response = _paginated_response([WorkItem.model_validate({"id": "wi-1", "name": "One", "target_date": None})])
    cycles = SimpleNamespace(list_work_items=lambda **kwargs: response)
    client = SimpleNamespace(cycles=cycles)
    monkeypatch.setattr("plane_mcp.tools.cycles.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(project_id="project", cycle_id="cycle", fields="id,name")

    assert result["results"] == [{"id": "wi-1", "name": "One"}]


def test_list_module_work_items_projects_each_result(monkeypatch):
    fn = _tool_fn(register_module_tools, "list_module_work_items")
    response = _paginated_response([WorkItem.model_validate({"id": "wi-1", "name": "One", "target_date": None})])
    modules = SimpleNamespace(list_work_items=lambda **kwargs: response)
    client = SimpleNamespace(modules=modules)
    monkeypatch.setattr("plane_mcp.tools.modules.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(project_id="project", module_id="module", fields="id,name")

    assert result["results"] == [{"id": "wi-1", "name": "One"}]


def test_retrieve_work_item_returns_sparse_dict_not_model(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "retrieve_work_item")
    work_item = WorkItemDetail.model_validate({"id": "wi-1", "name": "One", "target_date": None})
    work_items = SimpleNamespace(retrieve=lambda **kwargs: work_item)
    client = SimpleNamespace(work_items=work_items)
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(project_id="project", work_item_id="wi-1", fields="id,target_date")

    assert result == {"id": "wi-1", "target_date": None}
    assert isinstance(result, dict)
    assert not hasattr(result, "model_dump")


def test_retrieve_work_item_by_identifier_behaves_like_retrieve(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "retrieve_work_item_by_identifier")
    work_item = WorkItemDetail.model_validate({"id": "wi-1", "name": "One", "target_date": None})
    work_items = SimpleNamespace(retrieve_by_identifier=lambda **kwargs: work_item)
    client = SimpleNamespace(work_items=work_items)
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(work_item_identifier="PROJ-7", fields="id,target_date")

    assert result == {"id": "wi-1", "target_date": None}
    assert isinstance(result, dict)


def test_list_projects_honors_existing_fields_argument(monkeypatch):
    fn = _tool_fn(register_project_tools, "list_projects")
    response = SimpleNamespace(
        results=[Project.model_validate({"id": "project", "name": "Project", "identifier": "PROJ"})]
    )
    projects = SimpleNamespace(list=lambda **kwargs: response)
    client = SimpleNamespace(projects=projects)
    monkeypatch.setattr("plane_mcp.tools.projects.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(fields="id,name")

    assert result == [{"id": "project", "name": "Project"}]


def test_sdk_request_receives_original_fields_string(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "list_work_items")
    response = _paginated_response([WorkItem.model_validate({"id": "wi-1", "name": "One"})])
    seen = {}

    def list_work_items(**kwargs):
        seen["fields"] = kwargs["params"].fields
        return response

    work_items = SimpleNamespace(list=list_work_items, list_workspace=lambda **kwargs: response)
    client = SimpleNamespace(work_items=work_items)
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    fn(project_id="project", fields="id, name")

    assert seen["fields"] == "id, name"


def test_tool_schemas_still_expose_fields_arguments():
    async def schema_fields():
        mcp = FastMCP("test")
        register_work_item_tools(mcp)
        register_cycle_tools(mcp)
        register_module_tools(mcp)
        register_project_tools(mcp)
        list_work = await mcp.get_tool("list_work_items")
        retrieve = await mcp.get_tool("retrieve_work_item")
        by_identifier = await mcp.get_tool("retrieve_work_item_by_identifier")
        cycle = await mcp.get_tool("list_cycle_work_items")
        module = await mcp.get_tool("list_module_work_items")
        projects = await mcp.get_tool("list_projects")
        return [tool.parameters["properties"] for tool in [list_work, retrieve, by_identifier, cycle, module, projects]]

    for properties in asyncio.run(schema_fields()):
        assert "fields" in properties


def test_intake_list_and_retrieve_use_fields_from_existing_params(monkeypatch):
    list_fn = _tool_fn(register_intake_tools, "list_intake_work_items")
    retrieve_fn = _tool_fn(register_intake_tools, "retrieve_intake_work_item")
    intake_item = IntakeWorkItem.model_validate({"id": "intake-1", "issue": "wi-1", "source_email": None})
    seen = {}

    def list_intake(**kwargs):
        seen["list_fields"] = kwargs["params"].fields
        return SimpleNamespace(results=[intake_item])

    def retrieve_intake(**kwargs):
        seen["retrieve_fields"] = kwargs["params"].fields
        return intake_item

    client = SimpleNamespace(intake=SimpleNamespace(list=list_intake, retrieve=retrieve_intake))
    monkeypatch.setattr("plane_mcp.tools.intake.get_plane_client_context", lambda: (client, "workspace"))

    assert list_fn(project_id="project", params={"fields": "id,source_email"}) == [
        {"id": "intake-1", "source_email": None}
    ]
    assert retrieve_fn(project_id="project", work_item_id="wi-1", params={"fields": "id,source_email"}) == {
        "id": "intake-1",
        "source_email": None,
    }
    assert seen == {"list_fields": "id,source_email", "retrieve_fields": "id,source_email"}


def test_search_work_items_projects_each_issue(monkeypatch):
    fn = _tool_fn(register_work_item_tools, "search_work_items")
    search_item = WorkItemSearchItem(
        id="wi-1",
        name="One",
        sequence_id="1",
        project__identifier="PROJ",
        project_id="project",
        workspace__slug="workspace",
    )
    response = WorkItemSearch(issues=[search_item])
    work_items = SimpleNamespace(search=lambda **kwargs: response)
    client = SimpleNamespace(work_items=work_items)
    monkeypatch.setattr("plane_mcp.tools.work_items.get_plane_client_context", lambda: (client, "workspace"))

    result = fn(query="one", fields="id,name")

    assert result == {"issues": [{"id": "wi-1", "name": "One"}]}
