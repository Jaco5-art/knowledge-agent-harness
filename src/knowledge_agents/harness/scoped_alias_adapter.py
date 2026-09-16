
"""Document-scoped identity context; aliases are not relationship evidence."""
import hashlib
import json
from .adapters import OpenAIAdapter
from .contracts import EvidenceSpan, validate_evidence


class ScopedAliasAdapter(OpenAIAdapter):
    def __init__(self, model, embedding_model="text-embedding-3-small",
                 *, source, aliases, **kwargs):
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Nonempty source required")
        normalized = {}
        for alias, targets in aliases.items():
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("Invalid alias")
            targets = [targets] if isinstance(targets, str) else targets
            if not isinstance(targets, list) or not targets:
                raise ValueError("Alias must have one or more targets")
            if any(not isinstance(t, str) or not t.strip() for t in targets):
                raise ValueError("Invalid alias target")
            normalized[alias] = sorted(set(targets))

        self._source = source
        self._identity_json = json.dumps({
            "document_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "aliases": normalized,
        }, sort_keys=True, ensure_ascii=False)
        super().__init__(model, embedding_model, **kwargs)
        self.version += ":scoped-identity-v1:" + hashlib.sha256(
            self._identity_json.encode()
        ).hexdigest()

    async def _parse(self, prompt, payload, schema):
        data = json.loads(payload)
        if "source" in data and data["source"] != self._source:
            raise ValueError("Adapter cannot be reused for another document")

        if schema.__name__ == "EntityExtractionOutput":
            prompt += (
                " For this entity role, emit entity mentions only: "
                "subject is the entity surface name, predicate is 'mentioned in', "
                "object is 'document'. Preserve the surface name from the source. "
                "Do not emit partnership, investment, acquisition or other "
                "business relations as entity facts; another role handles them."
            )

        if schema.__name__ in ("VerifyOutput", "CorrectOutput"):
            evidence = [
                EvidenceSpan.model_validate(item)
                for item in data["evidence"]
            ]
            validate_evidence(self._source, evidence)
            data["trusted_identity_context"] = json.loads(self._identity_json)
            prompt += (
                " The application supplies trusted_identity_context separately "
                "from evidence. A single-target alias may establish entity "
                "identity only; it does not establish any relationship, date, "
                "direction or event completion. Multiple-target aliases remain "
                "ambiguous unless the evidence independently disambiguates them. "
                "Do not choose a target merely because it matches the candidate. "
                "All relationship claims must still follow from evidence spans. "
                "Cite only supplied evidence_ids, never identity metadata. "
                "Keep original quotes unchanged."
            )
            payload = json.dumps(data, ensure_ascii=False)

        return await super()._parse(prompt, payload, schema)
