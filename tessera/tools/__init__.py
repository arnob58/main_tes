"""Native tools: in-process capabilities that need live services (the model,
the GUI, vision). Each is registered with SkillRegistry.register_native() in
the composition root, so the orchestrator and pipeline never import them."""
