import asyncio
from types import SimpleNamespace

import pytest
from fastmcp import Client, FastMCP
from plane.errors.errors import HttpError

from plane_mcp.tools import projects as project_tools


class CapturingMCP:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator


class FakeProjects:
    def __init__(self, post_responses=None):
        self.post_responses = post_responses or {}
        self.post_calls = []
        self.patch_calls = []
        self.delete_calls = []
        self.get_members_calls = []

    def get_members(self, workspace_slug, project_id, params=None):
        self.get_members_calls.append((workspace_slug, project_id, params))
        return [
            {
                "id": "membership-1",
                "member": "user-1",
                "role": 15,
                "is_active": True,
            },
            {
                "id": "user-2",
                "email": "existing@example.com",
                "role": 20,
            },
        ]

    def _post(self, endpoint, data):
        self.post_calls.append((endpoint, data))
        response = self.post_responses.get(data["member"])
        if isinstance(response, Exception):
            raise response
        return response or {
            "id": f"membership-{data['member']}",
            "member": data["member"],
            "role": data["role"],
            "is_active": True,
        }

    def _patch(self, endpoint, data):
        self.patch_calls.append((endpoint, data))
        return {
            "id": "membership-1",
            "member": "user-1",
            "role": data["role"],
        }

    def _delete(self, endpoint):
        self.delete_calls.append(endpoint)
        return None


@pytest.fixture
def registered_tools():
    mcp = CapturingMCP()
    project_tools.register_project_tools(mcp)
    return mcp.tools


def patch_context(monkeypatch, fake_projects):
    client = SimpleNamespace(projects=fake_projects)
    monkeypatch.setattr(project_tools, "get_plane_client_context", lambda: (client, "workspace"))


def test_add_project_members_maps_member_id_to_public_endpoint(monkeypatch, registered_tools):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)

    result = registered_tools["add_project_members"](
        "project-1",
        [
            project_tools.ProjectMemberInput(member_id="user-1", role=15),
            project_tools.ProjectMemberInput(member_id="user-2", role=20),
        ],
    )

    assert fake_projects.post_calls == [
        ("workspace/projects/project-1/members", {"member": "user-1", "role": 15}),
        ("workspace/projects/project-1/members", {"member": "user-2", "role": 20}),
    ]
    assert [item.success for item in result] == [True, True]
    assert [item.member_id for item in result] == ["user-1", "user-2"]
    assert [item.membership_id for item in result] == ["membership-user-1", "membership-user-2"]


def test_add_project_members_rejects_duplicate_member_ids_before_api_calls(
    monkeypatch,
    registered_tools,
):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)

    with pytest.raises(ValueError, match="duplicate member_id"):
        registered_tools["add_project_members"](
            "project-1",
            [
                project_tools.ProjectMemberInput(member_id="user-1", role=15),
                project_tools.ProjectMemberInput(member_id="user-1", role=20),
            ],
        )

    assert fake_projects.post_calls == []


def test_add_project_members_rejects_empty_members_before_api_calls(monkeypatch, registered_tools):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)

    with pytest.raises(ValueError, match="at least one"):
        registered_tools["add_project_members"]("project-1", [])

    assert fake_projects.post_calls == []


def test_add_project_members_rejects_invalid_roles_before_api_calls(monkeypatch, registered_tools):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)
    invalid_member = project_tools.ProjectMemberInput.model_construct(member_id="user-1", role=10)

    with pytest.raises(ValueError, match="5 .*15 .*20"):
        registered_tools["add_project_members"]("project-1", [invalid_member])

    assert fake_projects.post_calls == []


def test_add_project_members_reports_partial_success_and_failure(monkeypatch, registered_tools):
    api_error = HttpError(
        "HTTP 400: Bad Request",
        400,
        {"member": ["User is already a project member."]},
    )
    fake_projects = FakeProjects(post_responses={"user-2": api_error})
    patch_context(monkeypatch, fake_projects)

    result = registered_tools["add_project_members"](
        "project-1",
        [
            project_tools.ProjectMemberInput(member_id="user-1", role=15),
            project_tools.ProjectMemberInput(member_id="user-2", role=20),
            project_tools.ProjectMemberInput(member_id="user-3", role=5),
        ],
    )

    assert [item.success for item in result] == [True, False, True]
    assert result[1].member_id == "user-2"
    assert result[1].status_code == 400
    assert "member: User is already a project member." in result[1].error
    assert result[1].response == {"member": ["User is already a project member."]}
    assert len(fake_projects.post_calls) == 3


def test_update_project_member_uses_membership_id_and_requested_payload(monkeypatch, registered_tools):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)

    result = registered_tools["update_project_member"]("project-1", "membership-1", 20)

    assert fake_projects.patch_calls == [
        ("workspace/projects/project-1/members/membership-1", {"role": 20})
    ]
    assert result.membership_id == "membership-1"
    assert result.member_id == "user-1"
    assert result.role == 20


def test_update_project_member_validates_role_before_api_call(monkeypatch, registered_tools):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)

    with pytest.raises(ValueError, match="5 .*15 .*20"):
        registered_tools["update_project_member"]("project-1", "membership-1", 99)

    assert fake_projects.patch_calls == []


def test_remove_project_member_uses_membership_id_and_returns_204_confirmation(
    monkeypatch,
    registered_tools,
):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)

    result = registered_tools["remove_project_member"]("project-1", "membership-1")

    assert fake_projects.delete_calls == ["workspace/projects/project-1/members/membership-1"]
    assert result == project_tools.ProjectMemberRemovalResult(
        membership_id="membership-1",
        removed=True,
    )


def test_project_memberships_expose_membership_and_user_ids_without_confusion(
    monkeypatch,
    registered_tools,
):
    fake_projects = FakeProjects()
    patch_context(monkeypatch, fake_projects)

    result = registered_tools["get_project_members"]("project-1", params={"expand": "member"})

    assert fake_projects.get_members_calls == [("workspace", "project-1", {"expand": "member"})]
    assert result[0].membership_id == "membership-1"
    assert result[0].member_id == "user-1"
    assert result[0].id == "user-1"
    assert result[1].membership_id is None
    assert result[1].member_id == "user-2"
    assert result[1].id == "user-2"


def test_project_membership_tools_are_registered_with_usable_fastmcp_schemas():
    async def list_project_tools():
        async with Client(mcp) as client:
            return {tool.name: tool for tool in await client.list_tools()}

    mcp = FastMCP("project-membership-test")
    project_tools.register_project_tools(mcp)

    tools = asyncio.run(list_project_tools())

    assert "add_project_members" in tools
    assert "update_project_member" in tools
    assert "remove_project_member" in tools

    add_schema = tools["add_project_members"].inputSchema
    update_schema = tools["update_project_member"].inputSchema
    remove_schema = tools["remove_project_member"].inputSchema

    assert set(add_schema["properties"]) >= {"project_id", "members"}
    member_item_schema = add_schema["properties"]["members"]["items"]
    assert set(member_item_schema["properties"]) == {"member_id", "role"}
    assert member_item_schema["properties"]["role"]["enum"] == [5, 15, 20]
    assert update_schema["properties"]["membership_id"]["type"] == "string"
    assert update_schema["properties"]["role"]["enum"] == [5, 15, 20]
    assert "is_active" not in update_schema["properties"]
    assert remove_schema["properties"]["membership_id"]["type"] == "string"
