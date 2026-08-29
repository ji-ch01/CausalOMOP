"""Study-companion conversation layer.

Everything in this package is an *advisory* layer over the governed workflow:
it records and renders the natural-language exchange between a reviewer and
the assistant, but nothing here can approve a gate, advance a phase, or
otherwise mutate `RunState`. Those transitions stay with
`causal.orchestrator.workflow.WorkflowManager` alone.
"""

from causal.companion.conversation import (
    CONVERSATION_FILENAME,
    ENTRY_KINDS,
    ENTRY_ROLES,
    ConversationEntry,
    ConversationStore,
    flush_intake,
)

__all__ = [
    "CONVERSATION_FILENAME",
    "ENTRY_KINDS",
    "ENTRY_ROLES",
    "ConversationEntry",
    "ConversationStore",
    "flush_intake",
]
