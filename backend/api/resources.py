"""
EduPath AI — Learning Resource API
Team KRIYA | Meta Hackathon 2026

Fetches, ranks, and serves real course links for each curriculum topic.
Supports resource interaction tracking (click logging), topic study
marking, and AI-generated topic summaries.
"""
import asyncio
import logging
from fastapi import APIRouter, HTTPException

from environment.models import (
    ResourceCard, TopicPageResponse, LinkClickRequest, MarkCompleteRequest
)
from environment.curriculum import TOPIC_GRAPH
from environment.student import student_manager
from ai.resource_fetcher import fetch_resources_async, fetch_alternative_resources_async
from ai.roadmap_generator import generate_topic_summary

router = APIRouter(prefix="/resources", tags=["resources"])
logger = logging.getLogger(__name__)


def _find_topic(topic_id: str):
    """Find a topic by ID, with fuzzy matching for human-readable names."""
    # 1. Exact match
    if topic_id in TOPIC_GRAPH:
        return TOPIC_GRAPH[topic_id]
    # 2. Normalized match (lowercase, underscored)
    normalized = topic_id.lower().replace(" ", "_").replace("-", "_")
    if normalized in TOPIC_GRAPH:
        return TOPIC_GRAPH[normalized]
    # 3. Partial name/id match
    for tid, topic in TOPIC_GRAPH.items():
        if normalized in tid or tid in normalized:
            return topic
        if normalized.replace("_", " ") in topic.name.lower():
            return topic
    # 4. Keyword match (e.g. "python" matches "python_basics")
    for tid, topic in TOPIC_GRAPH.items():
        if normalized.replace("_", "") in tid.replace("_", ""):
            return topic
    return None


def _topic_name_from_id(topic_id: str) -> str:
    """Convert a topic ID like 'flask_framework' to 'Flask Framework'."""
    return topic_id.replace("_", " ").replace("-", " ").title()


@router.get("/{topic_id}", response_model=TopicPageResponse)
async def get_topic_resources(topic_id: str, student_id: str = ""):
    """
    Get resources and AI summary for a topic.
    Works for ANY topic name — if not in the hardcoded graph, generates dynamically.
    """
    topic = _find_topic(topic_id)

    # Determine topic name and field
    if topic:
        topic_name = topic.name
        topic_field = topic.field
    else:
        # Dynamic topic — convert ID to readable name
        topic_name = _topic_name_from_id(topic_id)
        topic_field = "tech"  # default field
        logger.info(f"Dynamic topic resolution: '{topic_id}' → '{topic_name}'")

    # Get student context for personalized summary
    student = student_manager.get(student_id) if student_id else None
    field = student.target_field if student else topic_field
    goal = student.learning_goal if student else f"Learn {topic_name}"

    # Fetch resources via multi-platform smart search
    raw_resources = await fetch_resources_async(topic_name, topic_id)

    # Only return top 3 initially (rest available via /alternative endpoint)
    resources = [
        ResourceCard(
            title=r["title"],
            url=r["url"],
            source=r.get("source", "Other"),
            description=r.get("description", ""),
            duration_estimate=r.get("duration_estimate", "~2 hours"),
            resource_type=r.get("resource_type", "course"),
        )
        for r in raw_resources[:3]
    ]

    # Generate AI summary (works for ANY topic name)
    ai_summary = await generate_topic_summary(topic_name, field, goal)

    # Check student state
    can_mark_complete = False
    quiz_unlocked = False
    if student:
        # Check both the raw topic_id and normalized versions
        clicked = student.clicked_resource_links
        can_mark_complete = (
            (topic_id in clicked and len(clicked[topic_id]) > 0) or
            (topic_name.lower().replace(" ", "_") in clicked)
        )
        quiz_unlocked = (
            topic_id in student.topics_studied or
            topic_name.lower().replace(" ", "_") in student.topics_studied
        )

    return TopicPageResponse(
        topic_id=topic_id,
        topic_name=topic_name,
        ai_summary=ai_summary,
        resources=resources,
        can_mark_complete=can_mark_complete,
        quiz_unlocked=quiz_unlocked,
    )


@router.get("/{topic_id}/alternative")
async def get_alternative_resources(topic_id: str):
    """
    Get alternative course suggestions beyond the initial 3.
    Falls back to a Google search link when exhausted.
    """
    topic = _find_topic(topic_id)
    topic_name = topic.name if topic else _topic_name_from_id(topic_id)

    alt_resources = await fetch_alternative_resources_async(topic_name, topic_id, offset=3)

    return {
        "topic_id": topic_id,
        "topic_name": topic_name,
        "resources": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source": r.get("source", "Other"),
                "description": r.get("description", ""),
                "duration_estimate": r.get("duration_estimate", "~2 hours"),
                "resource_type": r.get("resource_type", "course"),
                "is_fallback": r.get("is_fallback", False),
                "quality_score": r.get("quality_score", 0),
            }
            for r in alt_resources
        ],
        "has_more": not any(r.get("is_fallback") for r in alt_resources),
    }


@router.post("/{topic_id}/link-clicked")
async def record_link_click(topic_id: str, body: LinkClickRequest):
    """
    Record that a student clicked a resource link.
    Unlocks the 'Mark as Complete' button.
    Works for any topic — no graph lookup needed.
    """
    student = student_manager.get(body.student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student '{body.student_id}' not found")

    is_first = student_manager.record_link_click(body.student_id, topic_id, body.resource_url)

    return {
        "can_mark_complete": True,
        "is_first_click": is_first,
        "topic_id": topic_id,
    }


@router.post("/{topic_id}/mark-complete")
async def mark_topic_complete(topic_id: str, body: MarkCompleteRequest):
    """
    Mark a topic as 'studied' (link clicked + student confirms).
    Only works if student has clicked at least one resource link.
    Works for any topic — no graph lookup needed.
    """
    student = student_manager.get(body.student_id)
    if not student:
        raise HTTPException(status_code=404, detail=f"Student '{body.student_id}' not found")

    success = student_manager.mark_topic_studied(body.student_id, topic_id)

    if not success:
        raise HTTPException(
            status_code=403,
            detail="Cannot mark complete: no resource link was clicked for this topic. "
                   "Please visit at least one course link first."
        )

    topic_name = _topic_name_from_id(topic_id)
    topic = _find_topic(topic_id)
    if topic:
        topic_name = topic.name

    return {
        "quiz_unlocked": True,
        "topic_id": topic_id,
        "message": f"Topic '{topic_name}' marked as studied. Quiz is now unlocked!",
    }
