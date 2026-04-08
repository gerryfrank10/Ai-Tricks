# AI Security

Security and AI intersect in two key ways: securing AI systems from attacks, and using AI to power defensive security tools.

---

## Two Sides of AI Security

```
┌─────────────────────────────────────┐
│  SECURING AI SYSTEMS                │
│  (AI as the target)                 │
│  • Adversarial examples             │
│  • Prompt injection                 │
│  • Model extraction/stealing        │
│  • Data poisoning & backdoors       │
│  • Membership inference             │
│  • Jailbreaks & safety bypasses     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  AI FOR CYBERSECURITY               │
│  (AI as the weapon/shield)          │
│  • Anomaly & intrusion detection    │
│  • Malware classification           │
│  • Log analysis with LLMs           │
│  • Threat intelligence automation   │
│  • Vulnerability triage             │
│  • AI-powered SIEM                  │
└─────────────────────────────────────┘
```

---

## Topics

| Page | Description |
|------|-------------|
| [AI Security & Red Teaming](ai-security.md) | Adversarial attacks, prompt injection, defenses, red teaming LLMs |
| [AI for Cybersecurity](cybersecurity.md) | UEBA, NIDS, log analysis, threat intel, AI-powered SIEM |

---

## Key Principles

1. **Defense in depth** — no single control is sufficient; layer input validation, rate limiting, output filtering, and monitoring
2. **Assume compromise** — design systems to limit blast radius when (not if) a component is compromised
3. **Least privilege for AI agents** — grant tools/permissions only what's needed for the task
4. **Audit trails** — every AI decision that triggers a response must be logged immutably
5. **Human in the loop for irreversible actions** — deleting, sending, paying → require approval

---

## Related Topics

- [Prompt Engineering Security](../llm/prompt-engineering.md)
- [AI Ethics](../industry/ethics.md)
- [MLOps & Deployment](../mlops/index.md)
