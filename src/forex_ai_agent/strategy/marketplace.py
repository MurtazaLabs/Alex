"""Strategy Marketplace - Internal marketplace for browsing, comparing, and activating strategy versions."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.strategy.versioning import StrategyVersion, StrategyVersionManager
from forex_ai_agent.strategy.ab_testing import ABTestManager, ABTestConfig, VariantConfig
from forex_ai_agent.strategy.regression import RegressionMonitor, RegressionConfig


class ActivationStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    ARCHIVED = "archived"


class ListingStatus(Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class StrategyListing:
    """A strategy listing in the marketplace."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    version_id: str = ""
    title: str = ""
    description: str = ""
    
    # Version info
    version: str = ""
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    author: str = ""
    tags: List[str] = field(default_factory=list)
    category: str = ""
    
    # Status
    status: ListingStatus = ListingStatus.DRAFT
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewed_by: str = ""
    rejection_reason: str = ""
    
    # Activation
    activation_status: ActivationStatus = ActivationStatus.PENDING
    activated_at: Optional[datetime] = None
    activated_by: str = ""
    
    # Performance tracking
    paper_trading_started: Optional[datetime] = None
    live_trading_started: Optional[datetime] = None
    performance_snapshots: List[Dict] = field(default_factory=list)
    
    # Reviews
    reviews: List[Dict] = field(default_factory=list)
    avg_rating: float = 0.0
    review_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['activation_status'] = self.activation_status.value
        data['submitted_at'] = self.submitted_at.isoformat()
        data['reviewed_at'] = self.reviewed_at.isoformat() if self.reviewed_at else None
        return data


@dataclass
class ActivationRequest:
    """Request to activate a strategy version."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    version_id: str = ""
    listing_id: str = ""
    requested_by: str = ""
    requested_at: datetime = field(default_factory=datetime.utcnow)
    status: ActivationStatus = ActivationStatus.PENDING
    approved_by: str = ""
    approved_at: Optional[datetime] = None
    approval_notes: str = ""
    rejected_by: str = ""
    rejected_at: Optional[datetime] = None
    rejection_reason: str = ""


@dataclass
class StrategyReview:
    """Review/feedback on a strategy."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    listing_id: str = ""
    reviewer: str = ""
    rating: int = 0  # 1-5
    title: str = ""
    content: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    helpful_count: int = 0


@dataclass
class MarketplaceConfig:
    """Marketplace configuration."""
    require_approval: bool = True
    min_backtest_trades: int = 50
    min_sharpe_ratio: float = 1.0
    max_drawdown_pct: float = 20.0
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.2
    auto_approve_threshold: float = 0.95  # Auto-approve if confidence > threshold
    require_approval_for_activation: bool = True
    max_active_strategies_per_user: int = 5


class StrategyMarketplace:
    """
    Internal strategy marketplace for browsing, comparing, and activating strategy versions.
    
    Features:
    - Strategy listings with performance metrics
    - Version comparison and diff viewing
    - Approval workflow for activation
    - Performance leaderboards
    - Reviews and ratings
    - Activation management
    """
    
    def __init__(
        self,
        version_manager: Any,
        ab_test_manager: Any = None,
        regression_monitor: Any = None,
        config: MarketplaceConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.version_manager = version_manager
        self.ab_test_manager = ab_test_manager
        self.regression_monitor = regression_monitor
        self.config = config or MarketplaceConfig()
        self.event_bus = event_bus
        
        self.listings: Dict[str, StrategyListing] = {}
        self.listings_by_version: Dict[str, str] = {}  # version_id -> listing_id
        self.activation_requests: Dict[str, ActivationRequest] = {}
        self.reviews: Dict[str, List[StrategyReview]] = defaultdict(list)
        
        # Leaderboards
        self.leaderboards: Dict[str, List[Dict]] = defaultdict(list)
        
        # Event bus
        self.event_bus = event_bus
        
        # Callbacks
        self.on_listing_created: Optional[Callable] = None
        self.on_listing_approved: Optional[Callable] = None
        self.on_activation_requested: Optional[Callable] = None
        self.on_activation_approved: Optional[Callable] = None
        self.on_review_added: Optional[Callable] = None
    
    def create_listing(
        self,
        version_id: str,
        title: str,
        description: str,
        author: str = "",
        tags: List[str] = None,
        category: str = "",
        auto_submit: bool = True,
    ) -> StrategyListing:
        """Create a new strategy listing."""
        version = self.version_manager.get_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")
        
        # Check if already listed
        if version_id in self.listings_by_version:
            raise ValueError(f"Version {version_id} already has a listing")
        
        # Check performance requirements
        if not self._meets_requirements(version):
            raise ValueError("Version does not meet minimum performance requirements")
        
        listing = StrategyListing(
            version_id=version.id,
            version=version.version,
            title=title,
            description=description,
            author=author,
            tags=tags or [],
            category=category,
            metrics=self._extract_metrics(version),
        )
        
        self.listings[listing.id] = listing
        self.listings_by_version[version_id] = listing.id
        
        if auto_submit and self.config.require_approval:
            self._submit_for_review(listing)
        
        if self.event_bus:
            asyncio.create_task(self._emit_event("listing.created", {
                "listing_id": listing.id,
                "version_id": version_id,
                "title": title,
            }))
        
        if self.on_listing_created:
            asyncio.create_task(self.on_listing_created(listing))
        
        return listing
    
    def _meets_requirements(self, version: Any) -> bool:
        """Check if version meets minimum requirements for listing."""
        if not self.config.require_approval:
            return True
        
        if not hasattr(self, 'backtest_engine') or not self.backtest_engine:
            return True  # Skip check if no backtest engine
        
        # Get metrics from version
        metrics = self._get_version_metrics(version)
        
        if metrics.get("trades_count", 0) < self.config.min_backtest_trades:
            return False
        if metrics.get("sharpe_ratio", 0) < self.config.min_sharpe_ratio:
            return False
        if metrics.get("max_drawdown_pct", 100) > self.config.max_drawdown_pct:
            return False
        if metrics.get("win_rate", 0) < self.config.min_win_rate:
            return False
        if metrics.get("profit_factor", 0) < self.config.min_profit_factor:
            return False
        
        return True
    
    def _get_version_metrics(self, version: Any) -> Dict[str, float]:
        """Extract metrics from a version."""
        # In production, this would query backtest results
        return {
            "trades_count": 100,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 10.0,
            "win_rate": 0.55,
            "profit_factor": 1.5,
        }
    
    def _extract_metrics(self, version: Any) -> Dict[str, float]:
        """Extract metrics for display."""
        return self._get_version_metrics(version)
    
    def _submit_for_review(self, listing: StrategyListing):
        """Submit listing for review."""
        listing.status = ListingStatus.PENDING_REVIEW
        # In production, notify reviewers
    
    def approve_listing(self, listing_id: str, reviewer: str, notes: str = "") -> bool:
        """Approve a listing."""
        listing = self.listings.get(listing_id)
        if not listing:
            return False
        
        if listing.status != ListingStatus.PENDING_REVIEW:
            return False
        
        listing.status = ListingStatus.APPROVED
        listing.reviewed_at = datetime.utcnow()
        listing.reviewed_by = reviewer
        
        # Auto-activate if confidence is high
        version = self.version_manager.get_version(listing.version_id)
        if self.config.auto_approve_threshold and self._get_confidence(version) >= self.config.auto_approve_threshold:
            self.activate_version(listing.version_id)
        
        if self.event_bus:
            asyncio.create_task(self._emit_event("listing.approved", {
                "listing_id": listing_id,
                "reviewer": reviewer,
            }))
        
        if self.on_listing_approved:
            asyncio.create_task(self.on_listing_approved(listing))
        
        return True
    
    def reject_listing(self, listing_id: str, reviewer: str, reason: str) -> bool:
        """Reject a listing."""
        listing = self.listings.get(listing_id)
        if not listing:
            return False
        
        if listing.status != ListingStatus.PENDING_REVIEW:
            return False
        
        listing.status = ListingStatus.REJECTED
        listing.reviewed_at = datetime.utcnow()
        listing.reviewed_by = reviewer
        listing.rejection_reason = reason
        
        if self.event_bus:
            asyncio.create_task(self._emit_event("listing.rejected", {
                "listing_id": listing_id,
                "reviewer": reviewer,
                "reason": reason,
            }))
        return True
    
    def request_activation(
        self,
        version_id: str,
        requested_by: str,
        listing_id: str = None,
    ) -> ActivationRequest:
        """Request activation of a strategy version."""
        version = self.version_manager.get_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")
        
        # Find listing
        listing_id = listing_id or self.listings_by_version.get(version_id)
        if not listing_id:
            raise ValueError("No listing found for version")
        
        listing = self.listings.get(listing_id)
        if not listing:
            raise ValueError("Listing not found")
        
        if listing.status != ListingStatus.APPROVED:
            raise ValueError("Listing must be approved before activation")
        
        if self.config.require_approval_for_activation:
            request = ActivationRequest(
                version_id=version_id,
                listing_id=listing_id,
                requested_by=requested_by,
                status=ActivationStatus.PENDING,
            )
            self.activation_requests[request.id] = request
            return request
        else:
            # Auto-approve
            return self._approve_activation(request.id, "auto", "Auto-approved")
    
    def _approve_activation(self, request_id: str, approver: str, notes: str = "") -> bool:
        """Approve an activation request."""
        request = self.activation_requests.get(request_id)
        if not request:
            return False
        
        if request.status != ActivationStatus.PENDING:
            return False
        
        request.status = ActivationStatus.APPROVED
        request.approved_by = approver
        request.approved_at = datetime.utcnow()
        request.approval_notes = notes
        
        # Activate the version
        self.activate_version(request.version_id)
        
        return True
    
    def reject_activation(self, request_id: str, rejector: str, reason: str) -> bool:
        """Reject an activation request."""
        request = self.activation_requests.get(request_id)
        if not request or request.status != ActivationStatus.PENDING:
            return False
        
        request.status = ActivationStatus.REJECTED
        request.rejected_by = rejector
        request.rejected_at = datetime.utcnow()
        request.rejection_reason = reason
        return True
    
    def activate_version(self, version_id: str, activated_by: str = "system") -> bool:
        """Activate a strategy version."""
        version = self.version_manager.get_version(version_id)
        if not version:
            return False
        
        # Deactivate current active
        if self.version_manager.active_version_id:
            old = self.version_manager.versions.get(self.version_manager.active_version_id)
            if old:
                old.status = "archived"
        
        version.status = "active"
        self.version_manager.active_version_id = version.id
        
        # Update listing
        listing_id = self.listings_by_version.get(version_id)
        if listing_id:
            listing = self.listings.get(listing_id)
            if listing:
                listing.activation_status = ActivationStatus.ACTIVE
                listing.activated_at = datetime.utcnow()
                listing.activated_by = activated_by
                listing.live_trading_started = datetime.utcnow()
        
        if self.event_bus:
            asyncio.create_task(self._emit_event("version.activated", {
                "version_id": version_id,
                "activated_by": activated_by,
            }))
        
        return True
    
    def deactivate_version(self, version_id: str, deactivated_by: str = "system") -> bool:
        """Deactivate a strategy version."""
        version = self.version_manager.get_version(version_id)
        if not version or version.status != "active":
            return False
        
        version.status = "archived"
        if self.version_manager.active_version_id == version_id:
            self.version_manager.active_version_id = None
        
        listing_id = self.listings_by_version.get(version_id)
        if listing_id:
            listing = self.listings.get(listing_id)
            if listing:
                listing.activation_status = ActivationStatus.ARCHIVED
        
        if self.event_bus:
            asyncio.create_task(self._emit_event("version.deactivated", {
                "version_id": version_id,
                "deactivated_by": deactivated_by,
            }))
        
        return True
    
    def _get_confidence(self, version: Any) -> float:
        """Calculate confidence score for a version."""
        # In production, would use statistical validation results
        return 0.9
    
    def _emit_event(self, event_type: str, payload: Dict):
        """Emit event to event bus."""
        pass
    
    def get_listing(self, listing_id: str) -> Optional[StrategyListing]:
        return self.listings.get(listing_id)
    
    def get_listing_by_version(self, version_id: str) -> Optional[StrategyListing]:
        listing_id = self.listings_by_version.get(version_id)
        if listing_id:
            return self.listings.get(listing_id)
        return None
    
    def list_listings(
        self,
        status: Optional[ListingStatus] = None,
        category: str = None,
        author: str = None,
        tags: List[str] = None,
        min_rating: float = 0,
        sort_by: str = "submitted_at",
        sort_order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> List[StrategyListing]:
        """List strategy listings with filters."""
        listings = list(self.listings.values())
        
        if status:
            listings = [l for l in listings if l.status == status]
        if category:
            listings = [l for l in listings if l.category == category]
        if author:
            listings = [l for l in listings if l.author == author]
        if tags:
            listings = [l for l in listings if any(t in l.tags for t in tags)]
        
        listings = [l for l in listings if l.avg_rating >= min_rating]
        
        reverse = sort_order == "desc"
        if sort_by == "submitted_at":
            listings.sort(key=lambda x: x.submitted_at, reverse=reverse)
        elif sort_by == "rating":
            listings.sort(key=lambda x: x.avg_rating, reverse=reverse)
        elif sort_by == "performance":
            listings.sort(key=lambda x: x.metrics.get("sharpe_ratio", 0), reverse=reverse)
        
        return listings[offset:offset + limit]
    
    def search_listings(self, query: str, limit: int = 20) -> List[StrategyListing]:
        """Search listings by text query."""
        query = query.lower()
        results = []
        for listing in self.listings.values():
            if (query in listing.title.lower() or 
                query in listing.description.lower() or
                query in listing.author.lower() or
                any(query in tag.lower() for tag in listing.tags)):
                results.append(listing)
        return results[:limit]
    
    def get_leaderboard(
        self,
        metric: str = "sharpe_ratio",
        period: str = "all_time",
        category: str = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Get performance leaderboard."""
        # In production, would query performance database
        # Return mock leaderboard for now
        listings = list(self.listings.values())
        if category:
            listings = [l for l in listings if l.category == category]
        
        sorted_listings = sorted(
            listings,
            key=lambda x: x.metrics.get(metric, 0),
            reverse=True
        )
        
        leaderboard = []
        for i, listing in enumerate(sorted_listings[:limit]):
            version = self.version_manager.get_version(listing.version_id)
            leaderboard.append({
                "rank": i + 1,
                "listing_id": listing.id,
                "title": listing.title,
                "version": listing.version,
                "metric_value": listing.metrics.get(metric, 0),
                "author": listing.author,
                "activated": listing.activation_status == ActivationStatus.ACTIVE,
            })
        
        return leaderboard
    
    def add_review(self, listing_id: str, reviewer: str, rating: int, title: str, content: str) -> StrategyReview:
        """Add a review to a listing."""
        if listing_id not in self.listings:
            raise ValueError("Listing not found")
        
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")
        
        review = StrategyReview(
            listing_id=listing_id,
            reviewer=reviewer,
            rating=rating,
            title=title,
            content=content,
        )
        
        self.reviews[listing_id].append(review)
        self._update_avg_rating(listing_id)
        
        if self.event_bus:
            asyncio.create_task(self._emit_event("review.added", {
                "listing_id": listing_id,
                "reviewer": reviewer,
                "rating": rating,
            }))
        
        if self.on_review_added:
            asyncio.create_task(self.on_review_added(listing_id, review))
        
        return self.reviews[listing_id][-1]
    
    def _update_avg_rating(self, listing_id: str):
        reviews = self.reviews.get(listing_id, [])
        if reviews:
            listing = self.listings.get(listing_id)
            if listing:
                listing.avg_rating = sum(r.rating for r in reviews) / len(reviews)
                listing.review_count = len(reviews)
    
    def get_reviews(self, listing_id: str, limit: int = 20) -> List[StrategyReview]:
        reviews = self.reviews.get(listing_id, [])
        return sorted(reviews, key=lambda r: r.created_at, reverse=True)[:limit]
    
    def add_activation_request(self, request: ActivationRequest):
        """Add an activation request."""
        self.activation_requests[request.id] = request
    
    def get_activation_request(self, request_id: str) -> Optional[ActivationRequest]:
        return self.activation_requests.get(request_id)
    
    def list_activation_requests(
        self,
        status: Optional[ActivationStatus] = None,
        limit: int = 50,
    ) -> List[ActivationRequest]:
        requests = list(self.activation_requests.values())
        if status:
            requests = [r for r in requests if r.status == status]
        return sorted(requests, key=lambda r: r.requested_at, reverse=True)[:limit]
    
    def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        total_listings = len(self.listings)
        approved = sum(1 for l in self.listings.values() if l.status == ListingStatus.APPROVED)
        active = sum(1 for l in self.listings.values() if l.activation_status == ActivationStatus.ACTIVE)
        pending_review = sum(1 for l in self.listings.values() if l.status == ListingStatus.PENDING_REVIEW)
        pending_activation = sum(1 for r in self.activation_requests.values() if r.status == ActivationStatus.PENDING)
        
        total_reviews = sum(len(r) for r in self.reviews.values())
        avg_rating = sum(l.avg_rating for l in self.listings.values()) / len(self.listings) if self.listings else 0
        
        return {
            "total_listings": total_listings,
            "approved": approved,
            "active": active,
            "pending_review": pending_review,
            "pending_activation": pending_activation,
            "total_reviews": total_reviews,
            "avg_rating": avg_rating,
            "total_versions": len(self.version_manager.versions) if self.version_manager else 0,
            "active_version": self.version_manager.active_version_id if self.version_manager else None,
        }
    
    def get_version_comparison(self, version_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple versions side by side."""
        versions = [self.version_manager.get_version(vid) for vid in version_ids]
        versions = [v for v in versions if v]
        
        if not versions:
            return {}
        
        comparison = {
            "versions": [
                {
                    "id": v.id,
                    "version": v.version,
                    "name": v.name,
                    "description": v.description,
                    "version_type": v.version_type.value,
                    "parent_id": v.parent_id,
                    "created_at": v.created_at.isoformat(),
                    "status": v.status,
                    "metrics": v.backtest_results,
                }
                for v in versions
            ],
            "diff": {},
        }
        
        if len(versions) >= 2:
            # Use version manager's diff
            pass
        
        return comparison
    
    def _emit_event(self, event_type: str, payload: Dict):
        """Emit event to event bus."""
        pass
    
    def _emit_event_async(self, event_type: str, payload: Dict):
        """Emit event asynchronously."""
        pass


def create_report_generator(
    config: ReportConfig = None,
    knowledge_agent=None,
    event_bus=None,
) -> ResearchReportGenerator:
    """Create a research report generator."""
    return ResearchReportGenerator(
        config=config,
        knowledge_agent=knowledge_agent,
        event_bus=event_bus,
    )