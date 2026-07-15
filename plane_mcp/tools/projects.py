"""Project-related tools for Plane MCP Server."""

from typing import Any, Literal, get_args

from fastmcp import FastMCP
from plane.errors.errors import HttpError
from plane.models.enums import TimezoneEnum
from plane.models.estimates import (
    CreateEstimate,
    CreateEstimatePoint,
    Estimate,
    EstimatePoint,
    UpdateEstimate,
    UpdateEstimatePoint,
)
from plane.models.projects import (
    CreateProject,
    PaginatedProjectResponse,
    Project,
    ProjectFeature,
    ProjectWorklogSummary,
    UpdateProject,
)
from plane.models.query_params import PaginatedQueryParams
from pydantic import BaseModel, ConfigDict, Field

from plane_mcp.client import get_plane_client_context

ProjectRole = Literal[5, 15, 20]
PROJECT_ROLE_LABELS: dict[int, str] = {
    5: "Guest",
    15: "Member",
    20: "Admin",
}
VALID_PROJECT_ROLES = set(PROJECT_ROLE_LABELS)


class ProjectMemberInput(BaseModel):
    """Input for adding an existing workspace user to a project."""

    member_id: str = Field(..., description="Workspace user UUID to add to the project.")
    role: ProjectRole = Field(..., description="Project role: 5=Guest, 15=Member, 20=Admin.")


class ProjectMembership(BaseModel):
    """Compact project-membership record exposed through MCP."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    id: str | None = Field(
        None,
        description="Backward-compatible alias for the workspace user UUID when available.",
    )
    membership_id: str | None = Field(None, description="ProjectMember record UUID.")
    member_id: str | None = Field(None, description="Workspace user UUID.")
    role: int | None = None
    role_slug: str | None = None
    is_active: bool | None = None
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    avatar: str | None = None
    avatar_url: str | None = None
    display_name: str | None = None


class ProjectMemberMutationResult(BaseModel):
    """Per-member result for bulk project-membership mutations."""

    model_config = ConfigDict(extra="allow")

    member_id: str
    success: bool
    role: int | None = None
    membership_id: str | None = None
    is_active: bool | None = None
    status_code: int | None = None
    error: str | None = None
    response: Any | None = None


class ProjectMemberRemovalResult(BaseModel):
    """Confirmation returned after removing a project membership."""

    membership_id: str
    removed: bool


class ProjectMembershipAPI:
    """Small bridge for project-member mutations missing from plane-sdk 0.2.16."""

    def __init__(self, projects_resource: Any) -> None:
        self._projects = projects_resource

    def list(
        self,
        workspace_slug: str,
        project_id: str,
        params: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self._projects.get_members(
            workspace_slug=workspace_slug,
            project_id=project_id,
            params=params,
        )

    def create(
        self,
        workspace_slug: str,
        project_id: str,
        member_id: str,
        role: int,
    ) -> Any:
        return self._projects._post(
            f"{workspace_slug}/projects/{project_id}/members",
            {"member": member_id, "role": role},
        )

    def update_role(
        self,
        workspace_slug: str,
        project_id: str,
        membership_id: str,
        role: int,
    ) -> Any:
        return self._projects._patch(
            f"{workspace_slug}/projects/{project_id}/members/{membership_id}",
            {"role": role},
        )

    def delete(
        self,
        workspace_slug: str,
        project_id: str,
        membership_id: str,
    ) -> None:
        self._projects._delete(f"{workspace_slug}/projects/{project_id}/members/{membership_id}")


def _validate_project_role(role: int) -> None:
    if role not in VALID_PROJECT_ROLES:
        raise ValueError("role must be one of 5 (Guest), 15 (Member), or 20 (Admin)")


def _coerce_member_input(member: ProjectMemberInput | dict[str, Any]) -> ProjectMemberInput:
    if isinstance(member, ProjectMemberInput):
        return member
    return ProjectMemberInput.model_validate(member)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict(exclude_none=True)
    return {}


def _get_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        nested_id = value.get("id")
        return str(nested_id) if nested_id is not None else None
    return None


def _extract_member_id(data: dict[str, Any]) -> str | None:
    for key in ("member_id", "member", "user_id", "user"):
        member_id = _get_id(data.get(key))
        if member_id:
            return member_id

    member_detail = data.get("member_detail") or data.get("user_detail")
    member_id = _get_id(member_detail)
    if member_id:
        return member_id

    return str(data["id"]) if data.get("id") is not None else None


def _extract_membership_id(data: dict[str, Any]) -> str | None:
    for key in ("membership_id", "project_member_id"):
        membership_id = data.get(key)
        if membership_id is not None:
            return str(membership_id)

    if any(key in data for key in ("member", "member_id", "user", "user_id", "member_detail", "user_detail")):
        return str(data["id"]) if data.get("id") is not None else None

    return None


def _normalize_project_membership(value: Any) -> ProjectMembership:
    data = _as_dict(value)
    member_id = _extract_member_id(data)
    membership_id = _extract_membership_id(data)

    member_detail = data.get("member_detail") or data.get("user_detail")
    if isinstance(member_detail, dict):
        for key in ("first_name", "last_name", "email", "avatar", "avatar_url", "display_name"):
            if key not in data and key in member_detail:
                data[key] = member_detail[key]

    data["member_id"] = member_id
    data["membership_id"] = membership_id
    data["id"] = member_id
    return ProjectMembership.model_validate(data)


def _http_error_response(response: object | None) -> object | None:
    if response is None or isinstance(response, (str, int, float, bool, dict, list)):
        return response
    return str(response)


def register_project_tools(mcp: FastMCP) -> None:
    """Register all project-related tools with the MCP server."""

    @mcp.tool()
    def list_projects(
        cursor: str | None = None,
        per_page: int | None = None,
        expand: str | None = None,
        fields: str | None = None,
        order_by: str | None = None,
    ) -> list[Project]:
        """
        List all projects in a workspace.

        Args:
            workspace_slug: The workspace slug identifier
            cursor: Pagination cursor for getting next set of results
            per_page: Number of results per page (1-100)
            expand: Comma-separated list of related fields to expand in response
            fields: Comma-separated list of fields to include in response
            order_by: Field to order results by. Prefix with '-' for descending order

        Returns:
            List of Project objects
        """
        client, workspace_slug = get_plane_client_context()

        params = PaginatedQueryParams(
            cursor=cursor,
            per_page=per_page,
            expand=expand,
            fields=fields,
            order_by=order_by,
        )

        response: PaginatedProjectResponse = client.projects.list(
            workspace_slug=workspace_slug,
            params=params,
        )

        return response.results

    @mcp.tool()
    def create_project(
        name: str,
        identifier: str,
        description: str | None = None,
        project_lead: str | None = None,
        default_assignee: str | None = None,
        emoji: str | None = None,
        cover_image: str | None = None,
        module_view: bool | None = None,
        cycle_view: bool | None = None,
        issue_views_view: bool | None = None,
        page_view: bool | None = None,
        intake_view: bool | None = None,
        guest_view_all_features: bool | None = None,
        archive_in: int | None = None,
        close_in: int | None = None,
        timezone: str | None = None,
        external_source: str | None = None,
        external_id: str | None = None,
        is_issue_type_enabled: bool | None = None,
    ) -> Project:
        """
        Create a new project.

        Args:
            workspace_slug: The workspace slug identifier
            name: Project name
            identifier: Project identifier (e.g., "MP" for "My Project")
            description: Project description
            project_lead: UUID of the project lead user
            default_assignee: UUID of the default assignee user
            emoji: Emoji for the project
            cover_image: Cover image URL or asset ID
            module_view: Enable module view
            cycle_view: Enable cycle view
            issue_views_view: Enable issue views view
            page_view: Enable page view
            intake_view: Enable intake view
            guest_view_all_features: Allow guests to view all features
            archive_in: Days until auto-archive
            close_in: Days until auto-close
            timezone: Project timezone
            external_source: External system source name
            external_id: External system identifier
            is_issue_type_enabled: Enable issue types

        Returns:
            Created Project object
        """
        client, workspace_slug = get_plane_client_context()

        # Validate timezone against allowed literal values
        validated_timezone: TimezoneEnum | None = (
            timezone if timezone in get_args(TimezoneEnum) else None  # type: ignore[assignment]
        )

        data = CreateProject(
            name=name,
            identifier=identifier,
            description=description,
            project_lead=project_lead,
            default_assignee=default_assignee,
            emoji=emoji,
            cover_image=cover_image,
            module_view=module_view,
            cycle_view=cycle_view,
            issue_views_view=issue_views_view,
            page_view=page_view,
            intake_view=intake_view,
            guest_view_all_features=guest_view_all_features,
            archive_in=archive_in,
            close_in=close_in,
            timezone=validated_timezone,
            external_source=external_source,
            external_id=external_id,
            is_issue_type_enabled=is_issue_type_enabled,
        )

        return client.projects.create(workspace_slug=workspace_slug, data=data)

    @mcp.tool()
    def retrieve_project(project_id: str) -> Project:
        """
        Retrieve a project by ID.

        Args:
            workspace_slug: The workspace slug identifier
            project_id: UUID of the project

        Returns:
            Project object
        """
        client, workspace_slug = get_plane_client_context()
        return client.projects.retrieve(workspace_slug=workspace_slug, project_id=project_id)

    @mcp.tool()
    def update_project(
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        project_lead: str | None = None,
        default_assignee: str | None = None,
        identifier: str | None = None,
        emoji: str | None = None,
        cover_image: str | None = None,
        network: int | None = None,
        module_view: bool | None = None,
        cycle_view: bool | None = None,
        issue_views_view: bool | None = None,
        page_view: bool | None = None,
        intake_view: bool | None = None,
        guest_view_all_features: bool | None = None,
        archive_in: int | None = None,
        close_in: int | None = None,
        timezone: str | None = None,
        external_source: str | None = None,
        external_id: str | None = None,
        is_issue_type_enabled: bool | None = None,
        is_time_tracking_enabled: bool | None = None,
        default_state: str | None = None,
        estimate: str | None = None,
    ) -> Project:
        """
        Update a project by ID.

        Args:
            workspace_slug: The workspace slug identifier
            project_id: UUID of the project
            name: Project name
            description: Project description
            project_lead: UUID of the project lead user
            default_assignee: UUID of the default assignee user
            identifier: Project identifier
            emoji: Emoji for the project
            cover_image: Cover image URL or asset ID
            network: Project visibility (0=secret, 2=public)
            module_view: Enable module view
            cycle_view: Enable cycle view
            issue_views_view: Enable issue views view
            page_view: Enable page view
            intake_view: Enable intake view
            guest_view_all_features: Allow guests to view all features
            archive_in: Days until auto-archive
            close_in: Days until auto-close
            timezone: Project timezone
            external_source: External system source name
            external_id: External system identifier
            is_issue_type_enabled: Enable issue types
            is_time_tracking_enabled: Enable time tracking
            default_state: UUID of the default state
            estimate: Estimate configuration

        Returns:
            Updated Project object
        """
        if network is not None and network not in {0, 2}:
            raise ValueError("network must be 0 (secret) or 2 (public)")

        client, workspace_slug = get_plane_client_context()

        # Validate timezone against allowed literal values
        validated_timezone: TimezoneEnum | None = (
            timezone if timezone in get_args(TimezoneEnum) else None  # type: ignore[assignment]
        )

        data = UpdateProject(
            name=name,
            description=description,
            project_lead=project_lead,
            default_assignee=default_assignee,
            identifier=identifier,
            emoji=emoji,
            cover_image=cover_image,
            network=network,
            module_view=module_view,
            cycle_view=cycle_view,
            issue_views_view=issue_views_view,
            page_view=page_view,
            intake_view=intake_view,
            guest_view_all_features=guest_view_all_features,
            archive_in=archive_in,
            close_in=close_in,
            timezone=validated_timezone,
            external_source=external_source,
            external_id=external_id,
            is_issue_type_enabled=is_issue_type_enabled,
            is_time_tracking_enabled=is_time_tracking_enabled,
            default_state=default_state,
            estimate=estimate,
        )

        return client.projects.update(workspace_slug=workspace_slug, project_id=project_id, data=data)

    @mcp.tool()
    def delete_project(project_id: str) -> None:
        """
        Delete a project by ID.

        Args:
            workspace_slug: The workspace slug identifier
            project_id: UUID of the project
        """
        client, workspace_slug = get_plane_client_context()
        client.projects.delete(workspace_slug=workspace_slug, project_id=project_id)

    @mcp.tool()
    def manage_project_archive(project_id: str, archive: bool) -> None:
        """
        Archive or unarchive a project.

        Archived projects are hidden from active project lists but not deleted.
        All work items, cycles, and modules are preserved.

        Args:
            project_id: UUID of the project
            archive: True to archive the project, False to unarchive it
        """
        client, workspace_slug = get_plane_client_context()
        if archive:
            client.projects.archive(workspace_slug=workspace_slug, project_id=project_id)
        else:
            client.projects.unarchive(workspace_slug=workspace_slug, project_id=project_id)

    @mcp.tool()
    def get_project_worklog_summary(project_id: str) -> list[ProjectWorklogSummary]:
        """
        Get work log summary for a project.

        Args:
            workspace_slug: The workspace slug identifier
            project_id: UUID of the project

        Returns:
            List of ProjectWorklogSummary objects containing work item IDs and durations
        """
        client, workspace_slug = get_plane_client_context()
        return client.projects.get_worklog_summary(workspace_slug=workspace_slug, project_id=project_id)

    @mcp.tool()
    def get_project_members(project_id: str, params: dict[str, Any] | None = None) -> list[ProjectMembership]:
        """
        Get all members of a project.

        The response includes both member_id (workspace user UUID) and
        membership_id (project-membership UUID, when returned by Plane). Use
        membership_id with update_project_member and remove_project_member.

        Args:
            workspace_slug: The workspace slug identifier
            project_id: UUID of the project
            params: Optional query parameters as a dictionary

        Returns:
            List of project membership records
        """
        client, workspace_slug = get_plane_client_context()
        memberships = ProjectMembershipAPI(client.projects).list(
            workspace_slug=workspace_slug,
            project_id=project_id,
            params=params,
        )
        return [_normalize_project_membership(member) for member in memberships]

    @mcp.tool()
    def add_project_members(
        project_id: str,
        members: list[ProjectMemberInput],
    ) -> list[ProjectMemberMutationResult]:
        """
        Add existing workspace users to a project.

        Args:
            project_id: UUID of the project
            members: Workspace users and roles to add. member_id is the
                workspace user UUID; role must be 5 (Guest), 15 (Member), or
                20 (Admin).

        Returns:
            One result per requested member. Plane validation and permission
            failures are returned per member so bulk additions can partially
            succeed.
        """
        if not members:
            raise ValueError("members must contain at least one project member to add")

        member_inputs = [_coerce_member_input(member) for member in members]
        duplicate_member_ids = {
            member.member_id
            for member in member_inputs
            if sum(1 for candidate in member_inputs if candidate.member_id == member.member_id) > 1
        }
        if duplicate_member_ids:
            duplicate_list = ", ".join(sorted(duplicate_member_ids))
            raise ValueError(f"duplicate member_id values are not allowed: {duplicate_list}")

        for member in member_inputs:
            _validate_project_role(member.role)

        client, workspace_slug = get_plane_client_context()
        memberships_api = ProjectMembershipAPI(client.projects)
        results: list[ProjectMemberMutationResult] = []

        for member in member_inputs:
            try:
                response = memberships_api.create(
                    workspace_slug=workspace_slug,
                    project_id=project_id,
                    member_id=member.member_id,
                    role=member.role,
                )
            except HttpError as error:
                results.append(
                    ProjectMemberMutationResult(
                        member_id=member.member_id,
                        success=False,
                        role=member.role,
                        status_code=error.status_code,
                        error=str(error),
                        response=_http_error_response(error.response),
                    )
                )
                continue

            membership = _normalize_project_membership(response)
            results.append(
                ProjectMemberMutationResult(
                    member_id=membership.member_id or member.member_id,
                    success=True,
                    role=membership.role if membership.role is not None else member.role,
                    membership_id=membership.membership_id,
                    is_active=membership.is_active,
                    status_code=201,
                )
            )

        return results

    @mcp.tool()
    def update_project_member(
        project_id: str,
        membership_id: str,
        role: ProjectRole,
    ) -> ProjectMembership:
        """
        Change a project membership's role.

        Args:
            project_id: UUID of the project
            membership_id: ProjectMember record UUID, not the workspace user UUID
            role: New project role: 5 (Guest), 15 (Member), or 20 (Admin)

        Returns:
            Updated project membership record
        """
        _validate_project_role(role)

        client, workspace_slug = get_plane_client_context()
        response = ProjectMembershipAPI(client.projects).update_role(
            workspace_slug=workspace_slug,
            project_id=project_id,
            membership_id=membership_id,
            role=role,
        )
        membership = _normalize_project_membership(response)
        membership.membership_id = membership.membership_id or membership_id
        membership.role = membership.role if membership.role is not None else role
        return membership

    @mcp.tool()
    def remove_project_member(project_id: str, membership_id: str) -> ProjectMemberRemovalResult:
        """
        Remove a project membership.

        Plane soft-deactivates the membership and returns HTTP 204.

        Args:
            project_id: UUID of the project
            membership_id: ProjectMember record UUID, not the workspace user UUID

        Returns:
            Compact removal confirmation
        """
        client, workspace_slug = get_plane_client_context()
        ProjectMembershipAPI(client.projects).delete(
            workspace_slug=workspace_slug,
            project_id=project_id,
            membership_id=membership_id,
        )
        return ProjectMemberRemovalResult(membership_id=membership_id, removed=True)

    @mcp.tool()
    def update_project_features(
        project_id: str,
        modules: bool | None = None,
        cycles: bool | None = None,
        views: bool | None = None,
        pages: bool | None = None,
        intakes: bool | None = None,
        work_item_types: bool | None = None,
    ) -> ProjectFeature:
        """
        Update features of a project.

        Args:
            workspace_slug: The workspace slug identifier
            project_id: UUID of the project
            modules: Enable/disable modules feature
            cycles: Enable/disable cycles feature
            views: Enable/disable views feature
            pages: Enable/disable pages feature
            intakes: Enable/disable intakes feature
            work_item_types: Enable/disable work item types feature

        Returns:
            Updated ProjectFeature object
        """
        client, workspace_slug = get_plane_client_context()

        data = ProjectFeature(
            modules=modules,
            cycles=cycles,
            views=views,
            pages=pages,
            intakes=intakes,
            work_item_types=work_item_types,
        )

        return client.projects.update_features(workspace_slug=workspace_slug, project_id=project_id, data=data)

    @mcp.tool()
    def get_project_estimate(project_id: str) -> Estimate:
        """
        Get the estimate configuration for a project.

        Returns the active estimate system including its ID, which is required
        to call list_project_estimate_points.

        Args:
            project_id: UUID of the project

        Returns:
            Estimate object with id, name, and type fields
        """
        client, workspace_slug = get_plane_client_context()
        return client.estimates.retrieve(workspace_slug=workspace_slug, project_id=project_id)

    @mcp.tool()
    def list_project_estimate_points(project_id: str, estimate_id: str) -> list[EstimatePoint]:
        """
        List all valid estimate points for a project.

        Use this to discover the available estimate point UUIDs before calling
        update_work_item with an estimate_point value. Each EstimatePoint has
        an id (UUID to pass to update_work_item) and a value (display label
        such as "1", "2", "3", "5", "8" or "XS", "S", "M", "L", "XL").

        Workflow:
            1. Call get_project_estimate to get the estimate_id
            2. Call list_project_estimate_points with that estimate_id
            3. Pick the EstimatePoint whose value matches the user's intent
            4. Pass that EstimatePoint.id to update_work_item(estimate_point=...)

        Args:
            project_id: UUID of the project
            estimate_id: UUID of the estimate (from get_project_estimate)

        Returns:
            List of EstimatePoint objects, each with id and value fields
        """
        client, workspace_slug = get_plane_client_context()
        return client.estimates.list_points(
            workspace_slug=workspace_slug,
            project_id=project_id,
            estimate_id=estimate_id,
        )

    @mcp.tool()
    def create_project_estimate(
        project_id: str,
        name: str,
        type: str | None = None,
        description: str | None = None,
        last_used: bool = True,
        external_id: str | None = None,
        external_source: str | None = None,
    ) -> Estimate:
        """
        Create a new estimate for a project.

        Args:
            project_id: UUID of the project
            name: Name of the estimate (e.g., "Story Points", "T-Shirt Sizes")
            type: Estimate type — "categories", "points", or "time"
            description: Optional description
            last_used: Whether this becomes the active estimate (default True)
            external_id: External system identifier
            external_source: External system source name

        Returns:
            Created Estimate object
        """
        client, workspace_slug = get_plane_client_context()
        data = CreateEstimate(
            name=name,
            type=type,
            description=description,
            last_used=last_used,
            external_id=external_id,
            external_source=external_source,
        )
        return client.estimates.create(workspace_slug=workspace_slug, project_id=project_id, data=data)

    @mcp.tool()
    def update_project_estimate(
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        external_id: str | None = None,
        external_source: str | None = None,
    ) -> Estimate:
        """
        Update the estimate for a project.

        Args:
            project_id: UUID of the project
            name: New name for the estimate
            description: New description
            external_id: External system identifier
            external_source: External system source name

        Returns:
            Updated Estimate object
        """
        client, workspace_slug = get_plane_client_context()
        data = UpdateEstimate(
            name=name,
            description=description,
            external_id=external_id,
            external_source=external_source,
        )
        return client.estimates.update(workspace_slug=workspace_slug, project_id=project_id, data=data)

    @mcp.tool()
    def delete_project_estimate(project_id: str) -> None:
        """
        Delete the estimate for a project.

        Args:
            project_id: UUID of the project
        """
        client, workspace_slug = get_plane_client_context()
        client.estimates.delete(workspace_slug=workspace_slug, project_id=project_id)

    @mcp.tool()
    def link_estimate_to_project(project_id: str, estimate_id: str) -> Project:
        """
        Link an estimate to a project, making it the active estimate system.

        Args:
            project_id: UUID of the project
            estimate_id: UUID of the estimate to activate

        Returns:
            Updated Project object
        """
        client, workspace_slug = get_plane_client_context()
        return client.estimates.link_to_project(
            workspace_slug=workspace_slug,
            project_id=project_id,
            estimate_id=estimate_id,
        )

    @mcp.tool()
    def create_project_estimate_points(
        project_id: str,
        estimate_id: str,
        points: list[dict],
    ) -> list[EstimatePoint]:
        """
        Create estimate points for a project estimate.

        Each point dict may have: value (required, max 20 chars), key (int),
        description, external_id, external_source.

        Example:
            points=[
                {"value": "1", "key": 0},
                {"value": "2", "key": 1},
                {"value": "3", "key": 2},
                {"value": "5", "key": 3},
                {"value": "8", "key": 4},
            ]

        Args:
            project_id: UUID of the project
            estimate_id: UUID of the estimate
            points: List of point definitions

        Returns:
            List of created EstimatePoint objects
        """
        client, workspace_slug = get_plane_client_context()
        data = [CreateEstimatePoint(**p) for p in points]
        return client.estimates.create_points(
            workspace_slug=workspace_slug,
            project_id=project_id,
            estimate_id=estimate_id,
            data=data,
        )

    @mcp.tool()
    def update_project_estimate_point(
        project_id: str,
        estimate_id: str,
        estimate_point_id: str,
        value: str | None = None,
        key: int | None = None,
        description: str | None = None,
        external_id: str | None = None,
        external_source: str | None = None,
    ) -> EstimatePoint:
        """
        Update a single estimate point.

        Args:
            project_id: UUID of the project
            estimate_id: UUID of the estimate
            estimate_point_id: UUID of the estimate point to update
            value: New display value (max 20 chars, e.g. "XL", "13")
            key: New sort key (integer)
            description: New description
            external_id: External system identifier
            external_source: External system source name

        Returns:
            Updated EstimatePoint object
        """
        client, workspace_slug = get_plane_client_context()
        data = UpdateEstimatePoint(
            value=value,
            key=key,
            description=description,
            external_id=external_id,
            external_source=external_source,
        )
        return client.estimates.update_point(
            workspace_slug=workspace_slug,
            project_id=project_id,
            estimate_id=estimate_id,
            estimate_point_id=estimate_point_id,
            data=data,
        )

    @mcp.tool()
    def delete_project_estimate_point(
        project_id: str,
        estimate_id: str,
        estimate_point_id: str,
    ) -> None:
        """
        Delete a single estimate point.

        Args:
            project_id: UUID of the project
            estimate_id: UUID of the estimate
            estimate_point_id: UUID of the estimate point to delete
        """
        client, workspace_slug = get_plane_client_context()
        client.estimates.delete_point(
            workspace_slug=workspace_slug,
            project_id=project_id,
            estimate_id=estimate_id,
            estimate_point_id=estimate_point_id,
        )
