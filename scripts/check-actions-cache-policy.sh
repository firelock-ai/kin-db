#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workflow_root="${1:-${root}/.github/workflows}"
action_root="${2:-$(dirname "${workflow_root}")/actions}"

ruby - "${workflow_root}" "${action_root}" <<'RUBY'
require "digest"
require "psych"
require "set"
require "yaml"

workflow_root = File.expand_path(ARGV.fetch(0))
action_root = File.expand_path(ARGV.fetch(1))
abort("FAIL: workflow directory does not exist: #{workflow_root}") unless Dir.exist?(workflow_root)

expected_counts = {
  "ci.yml" => [2, 1],
  "ci-linux.yml" => [1, 0],
  "windows-nightly.yml" => [1, 1],
}.freeze
allowed_paths = ["~/.cargo/registry", "~/.cargo/git"].freeze
restore_key = "${{ runner.os }}-cargo-sources-v2"
restore_prefix = "${{ runner.os }}-cargo-sources-"
save_key = "${{ steps.cargo-sources.outputs.cache-primary-key }}"
save_condition = "github.ref == 'refs/heads/main' && steps.cargo-sources.outputs.cache-hit != 'true'"
fetch_run = ["set -euo pipefail", "cargo fetch"].freeze
guard_run = [
  "./scripts/check-actions-cache-policy.sh",
  "./scripts/test-actions-cache-policy.sh",
].freeze
approved_step_actions = Set.new([
  "./.github/actions/rust-toolchain",
  "./kin-db/.github/actions/rust-toolchain",
  "EmbarkStudios/cargo-deny-action@v2",
  "actions/cache/restore@v6",
  "actions/cache/save@v6",
  "actions/checkout@v7",
  "actions/upload-artifact@v4",
  "codecov/codecov-action@v5",
  "softprops/action-gh-release@v2",
  "taiki-e/install-action@v2",
]).freeze
approved_reusable_jobs = {
  ["kin-dependency-wave.yml", "dependency-wave"] =>
    "firelock-ai/kin-actions/.github/workflows/cargo-dependency-wave.yml@v0.1.31",
  ["registry-publish.yml", "release"] =>
    "firelock-ai/kin-actions/.github/workflows/cargo-registry-release.yml@v0.1.31",
  ["scheduled-failure-alarm.yml", "alarm"] =>
    "firelock-ai/kin-actions/.github/workflows/scheduled-failure-alarm.yml@v0.1.33",
}.freeze
owner_job_names = {
  "ci.yml" => "check",
  "windows-nightly.yml" => "windows",
}.freeze
owner_job_keys = {
  "ci.yml" => %w[name runs-on steps strategy],
  "windows-nightly.yml" => %w[name runs-on steps],
}.freeze
owner_step_topology = {
  "ci.yml" => [
    "@actions/checkout@v7",
    "Install Rust toolchain",
    "Restore cargo sources",
    "Check formatting",
    "Clippy",
    "Build",
    "Test",
    "100K hydration smoke guard",
    "Check the nightly Windows job still runs these steps",
    "Fetch complete Cargo source graph",
    "Save cargo sources on main",
  ],
  "windows-nightly.yml" => [
    "@actions/checkout@v7",
    "Install Rust toolchain",
    "Restore cargo sources",
    "Check formatting",
    "Clippy",
    "Build",
    "Test",
    "100K hydration smoke guard",
    "Check the nightly Windows job still runs these steps",
    "Fetch complete Cargo source graph",
    "Save cargo sources on main",
  ],
}.freeze
protected_run_contracts = {
  "Check formatting" => ["edafbe150671a93865d25affda0ceb8125b50863752550e2ba8a18ef69b2841e", %w[name run]],
  "Clippy" => ["786046eba8a6588787b01c2913e5ffb5482101680f91276f03a938168f6ca556", %w[name run shell]],
  "Build" => ["6525cb3546379952d414d046f14f9bd255b680be01de84658c2933d241252d74", %w[name run]],
  "Test" => ["4cb9ee46cf47c2359a046708d4e9f6599013360101992f749388bb9779c0fdc5", %w[env name run]],
  "100K hydration smoke guard" => ["27da212cc355a564550d8195ec7a2cc448b7f60a7193adfc69c1e5850e94a222", %w[env name run]],
  "Check the nightly Windows job still runs these steps" => ["3a04b48a4db5699f5b6fa9e0bbc66216ca3ffa810e2b2752ea30193a9f7787be", %w[if name run]],
  "Fetch complete Cargo source graph" => ["2846693bc6bcf6b18c86af1b1a8ae419a361a1da74620336af5ee673a08df1dc", %w[name run shell]],
}.freeze
protected_env_keys = Set.new(%w[CARGO_HOME CARGO_TARGET_DIR HOME]).freeze

errors = []
counts = {}
reusable_seen = Set.new
workflows = Dir[File.join(workflow_root, "*.{yml,yaml}")].sort
abort("FAIL: no workflow files found under #{workflow_root}") if workflows.empty?

def inspect_yaml_node(node, file_name, errors)
  case node
  when Psych::Nodes::Alias
    errors << "#{file_name}:#{node.start_line + 1}: YAML aliases are forbidden in workflow policy"
  when Psych::Nodes::Mapping
    seen = {}
    node.children.each_slice(2) do |key_node, value_node|
      unless key_node.is_a?(Psych::Nodes::Scalar)
        errors << "#{file_name}:#{key_node.start_line + 1}: complex YAML mapping keys are forbidden"
        inspect_yaml_node(value_node, file_name, errors)
        next
      end

      key = key_node.value
      if seen.key?(key)
        errors << (
          "#{file_name}:#{key_node.start_line + 1}: duplicate YAML mapping key #{key.inspect}; " \
          "first declared at line #{seen.fetch(key)}"
        )
      else
        seen[key] = key_node.start_line + 1
      end
      inspect_yaml_node(value_node, file_name, errors)
    end
  else
    Array(node.children).each { |child| inspect_yaml_node(child, file_name, errors) }
  end
end

def lines(value)
  return nil unless value.is_a?(String)

  value.lines.map(&:strip).reject(&:empty?)
end

def each_mapping(value, &block)
  case value
  when Hash
    yield(value)
    value.each_value { |child| each_mapping(child, &block) }
  when Array
    value.each { |child| each_mapping(child, &block) }
  end
end

def step_identity(step)
  step["name"] || "@#{step["uses"]}"
end

def inspect_protected_env(mapping, location, protected_env_keys, errors)
  env = mapping["env"]
  return unless env.is_a?(Hash)

  env.each_key do |key|
    if protected_env_keys.include?(key.to_s.upcase)
      errors << "#{location}: protected runner environment #{key} must not be overridden"
    end
  end
end

def false_value?(value)
  value == false || value.to_s.strip.downcase == "false"
end

def true_value?(value)
  value == true || value.to_s.strip.downcase == "true"
end

def inspect_hidden_cache_action(mapping, location, errors)
  action = mapping["uses"]
  return unless action.is_a?(String)

  normalized = action.downcase
  return if normalized.start_with?("actions/cache")

  inputs = mapping["with"].is_a?(Hash) ? mapping["with"] : {}
  cache_inputs = %w[
    cache
    cache-dependency-path
    cache-from
    cache-to
    cache-key
    shared-key
    save-cache
    restore-cache
  ]
  cache_inputs.each do |input|
    next unless inputs.key?(input)
    next if false_value?(inputs[input])

    errors << "#{location}: hidden cache input #{input.inspect} is forbidden"
  end

  action_name = normalized.split("@", 2).first
  case action_name
  when "actions/setup-node"
    unless inputs.key?("package-manager-cache") && false_value?(inputs["package-manager-cache"])
      errors << (
        "#{location}: actions/setup-node must set package-manager-cache: false " \
        "to disable its implicit dependency cache"
      )
    end
  when "actions/setup-go"
    unless inputs.key?("cache") && false_value?(inputs["cache"])
      errors << "#{location}: actions/setup-go must set cache: false"
    end
  when "gradle/actions/setup-gradle"
    unless inputs.key?("cache-disabled") && true_value?(inputs["cache-disabled"])
      errors << "#{location}: gradle/actions/setup-gradle must set cache-disabled: true"
    end
  when "astral-sh/setup-uv"
    unless inputs.key?("enable-cache") && false_value?(inputs["enable-cache"])
      errors << "#{location}: astral-sh/setup-uv must set enable-cache: false"
    end
  end

  if action_name.match?(%r{(?:^|/)(?:rust-cache|sccache-action|ccache-action|cache-apt-pkgs-action)$})
    errors << "#{location}: dedicated cache action #{action.inspect} is forbidden"
  end
end

workflows.each do |workflow|
  file_name = File.basename(workflow)
  content = File.read(workflow, encoding: "UTF-8")

  begin
    syntax_tree = Psych.parse_stream(content, filename: workflow)
    inspect_yaml_node(syntax_tree, file_name, errors)
    document = YAML.safe_load(
      content,
      permitted_classes: [],
      permitted_symbols: [],
      aliases: false,
      filename: workflow,
    )
  rescue Psych::Exception => error
    errors << "#{file_name}: YAML parse failed: #{error.message}"
    counts[file_name] = [0, 0]
    next
  end

  jobs = document.is_a?(Hash) ? document["jobs"] : nil
  unless jobs.is_a?(Hash)
    errors << "#{file_name}: jobs must be a YAML mapping"
    counts[file_name] = [0, 0]
    next
  end

  if document.key?("defaults")
    errors << "#{file_name}: workflow defaults are forbidden because they can mask guard failures"
  end

  each_mapping(document) do |mapping|
    inspect_protected_env(mapping, file_name, protected_env_keys, errors)
  end

  if file_name == "ci.yml"
    guard_job = jobs["schema-provenance"]
    unless guard_job.is_a?(Hash)
      errors << "ci.yml: required schema-provenance job is missing"
    else
      errors << "ci.yml: schema-provenance job name must remain Schema Provenance" unless guard_job["name"] == "Schema Provenance"
      errors << "ci.yml: schema-provenance must run on ubuntu-latest" unless guard_job["runs-on"] == "ubuntu-latest"
      unless guard_job.keys.sort == %w[name runs-on steps]
        errors << "ci.yml: schema-provenance job permits only name, runs-on, and steps"
      end

      guard_steps = guard_job["steps"]
      if !guard_steps.is_a?(Array) || guard_steps.length < 2
        errors << "ci.yml: schema-provenance must check out the repo and run the cache guard"
      else
        checkout = guard_steps[0]
        unless checkout.is_a?(Hash) && checkout.keys == ["uses"] &&
               checkout["uses"] == "actions/checkout@v7"
          errors << "ci.yml: schema-provenance must begin with an exact current-ref actions/checkout@v7"
        end

        guard = guard_steps[1]
        unless guard.is_a?(Hash) && guard["name"] == "Check Actions cache policy" &&
               guard["shell"] == "bash" && lines(guard["run"]) == guard_run &&
               guard.keys.sort == %w[name run shell]
          errors << "ci.yml: cache policy guard must be the first exact fail-hard step after checkout"
        end
      end
    end
  end

  restore_count = 0
  save_count = 0

  jobs.each do |job_name, job|
    next unless job.is_a?(Hash)

    reusable = job["uses"]
    if reusable
      identity = [file_name, job_name]
      expected = approved_reusable_jobs[identity]
      if expected != reusable
        errors << "#{file_name}: job #{job_name.inspect} uses unapproved reusable workflow #{reusable.inspect}"
      else
        reusable_seen << identity
      end
      next
    end

    steps = job["steps"]
    next if steps.nil?
    unless steps.is_a?(Array)
      errors << "#{file_name}: job #{job_name.inspect} steps must be a YAML sequence"
      next
    end

    if owner_job_names[file_name] == job_name
      expected_keys = owner_job_keys.fetch(file_name)
      unless job.keys.sort == expected_keys
        errors << "#{file_name}: cache-owner job permits only #{expected_keys.inspect}"
      end
      actual_topology = steps.map { |step| step.is_a?(Hash) ? step_identity(step) : "<invalid>" }
      unless actual_topology == owner_step_topology.fetch(file_name)
        errors << "#{file_name}: cache-owner step topology drifted"
      end

      protected_run_contracts.each do |step_name, (expected_digest, expected_keys_for_step)|
        matches = steps.select { |step| step.is_a?(Hash) && step["name"] == step_name }
        if matches.length != 1
          errors << "#{file_name}: cache-owner job requires exactly one #{step_name.inspect} step"
          next
        end
        protected = matches.first
        actual_digest = protected["run"].is_a?(String) ? Digest::SHA256.hexdigest(protected["run"]) : nil
        unless actual_digest == expected_digest && protected.keys.sort == expected_keys_for_step
          errors << "#{file_name}: protected #{step_name.inspect} step drifted or became non-authoritative"
        end
      end

      checkout = steps.first
      unless checkout.is_a?(Hash) && checkout.keys == ["uses"] && checkout["uses"] == "actions/checkout@v7"
        errors << "#{file_name}: cache-owner job must begin with an exact current-ref checkout"
      end
      steps.each do |step|
        next unless step.is_a?(Hash)
        if step.key?("continue-on-error")
          errors << "#{file_name}: cache-owner steps must not continue on error"
        end
        next unless step.key?("if")
        next if step["name"] == "Check the nightly Windows job still runs these steps" &&
                step["if"] == "runner.os == 'Linux'"
        next if step["name"] == "Save cargo sources on main" && step["if"] == save_condition

        errors << "#{file_name}: cache-owner step #{step_identity(step).inspect} has an unapproved condition"
      end
    end

    steps.each_with_index do |step, index|
      next unless step.is_a?(Hash)

      location = "#{file_name}: job #{job_name.inspect} step #{index + 1}"
      inspect_protected_env(step, location, protected_env_keys, errors)
      inspect_hidden_cache_action(step, location, errors)
      action = step["uses"]
      if action && (!action.is_a?(String) || !approved_step_actions.include?(action))
        errors << "#{location}: unapproved action identity #{action.inspect}"
      end
      next unless action.is_a?(String) && action.downcase.start_with?("actions/cache")

      cache_paths = lines(step.dig("with", "path")) if step["with"].is_a?(Hash)
      key = step.dig("with", "key") if step["with"].is_a?(Hash)
      body = step.inspect

      case action
      when "actions/cache/restore@v6"
        restore_count += 1
        unless step.keys.sort == %w[id name uses with]
          errors << "#{location}: cache restore step permits only id, name, uses, and with"
        end
        expected_inputs = %w[key path restore-keys]
        actual_inputs = step["with"].is_a?(Hash) ? step["with"].keys.sort : []
        if actual_inputs != expected_inputs
          errors << "#{location}: restore inputs must be exactly #{expected_inputs.inspect}"
        end
        errors << "#{location}: restore id must be cargo-sources" unless step["id"] == "cargo-sources"
        errors << "#{location}: cache restore must not continue on error" if step.key?("continue-on-error")
        if step.key?("if")
          errors << "#{location}: cache restore must run on every workflow ref"
        end
        if steps.take(index).any? { |prior| prior.is_a?(Hash) && prior.key?("run") }
          errors << "#{location}: cache restore must precede every run step"
        end
        if key != restore_key
          errors << "#{location}: restore key must be the bounded epoch #{restore_key}"
        end
        restore_keys = lines(step.dig("with", "restore-keys")) if step["with"].is_a?(Hash)
        if restore_keys != [restore_prefix]
          errors << "#{location}: restore prefix must be #{restore_prefix}"
        end
      when "actions/cache/save@v6"
        save_count += 1
        unless step.keys.sort == %w[if name uses with]
          errors << "#{location}: cache save step permits only if, name, uses, and with"
        end
        expected_inputs = %w[key path]
        actual_inputs = step["with"].is_a?(Hash) ? step["with"].keys.sort : []
        if actual_inputs != expected_inputs
          errors << "#{location}: save inputs must be exactly #{expected_inputs.inspect}"
        end
        errors << "#{location}: cache save must not continue on error" if step.key?("continue-on-error")
        if step["if"] != save_condition
          errors << "#{location}: cache save must be restricted to a main cache miss"
        end
        if key != save_key
          errors << "#{location}: save key must come from the restore primary key"
        end
        if index != steps.length - 1
          errors << "#{location}: cache save must be the last declared job step"
        end
        unless steps.take(index).any? do |prior|
                 prior.is_a?(Hash) && prior["id"] == "cargo-sources" &&
                   prior["uses"] == "actions/cache/restore@v6"
               end
          errors << "#{location}: cache save must follow cargo-sources restore in the same job"
        end
        fetch = steps[index - 1]
        unless fetch.is_a?(Hash) && fetch["name"] == "Fetch complete Cargo source graph" &&
               fetch["shell"] == "bash" && lines(fetch["run"]) == fetch_run &&
               fetch.keys.sort == %w[name run shell]
          errors << (
            "#{location}: cache save must immediately follow an unconditional fail-hard cargo fetch"
          )
        end
      else
        errors << (
          "#{location}: use actions/cache/restore@v6 or actions/cache/save@v6, not #{action}"
        )
      end

      if body.include?("hashFiles(") || body.include?("github.sha") || body.include?("github.run_id")
        errors << "#{location}: cache keys must not expand per dependency hash, SHA, or run"
      end
      if cache_paths != allowed_paths
        errors << (
          "#{location}: cache paths must be exactly #{allowed_paths.inspect}; " \
          "target output is forbidden"
        )
      end
      if cache_paths&.any? { |path| path.split("/").include?("target") }
        errors << "#{location}: target output is forbidden in Actions caches"
      end
    end
  end

  counts[file_name] = [restore_count, save_count]
end


approved_reusable_jobs.each_key do |identity|
  unless reusable_seen.include?(identity)
    errors << "#{identity[0]}: required reviewed reusable job #{identity[1].inspect} is missing"
  end
end

Dir[File.join(action_root, "**", "*.{yml,yaml}")].sort.each do |action_file|
  relative_name = action_file.delete_prefix("#{action_root}/")
  content = File.read(action_file, encoding: "UTF-8")

  begin
    syntax_tree = Psych.parse_stream(content, filename: action_file)
    inspect_yaml_node(syntax_tree, relative_name, errors)
    document = YAML.safe_load(
      content,
      permitted_classes: [],
      permitted_symbols: [],
      aliases: false,
      filename: action_file,
    )
  rescue Psych::Exception => error
    errors << "#{relative_name}: YAML parse failed: #{error.message}"
    next
  end

  each_mapping(document) do |mapping|
    inspect_protected_env(mapping, relative_name, protected_env_keys, errors)
    inspect_hidden_cache_action(mapping, relative_name, errors)
    action = mapping["uses"]
    if action && (!action.is_a?(String) || !approved_step_actions.include?(action))
      errors << "#{relative_name}: unapproved composite action identity #{action.inspect}"
    end
    next unless action.is_a?(String) && action.downcase.start_with?("actions/cache")

    errors << (
      "#{relative_name}: repo-local composite actions must not invoke actions/cache; " \
      "declare bounded caches in an audited workflow job"
    )
  end
end

expected_counts.each do |workflow_name, expected|
  actual = counts.fetch(workflow_name, [0, 0])
  next if actual == expected

  errors << (
    "#{workflow_name}: expected #{expected[0]} restore and #{expected[1]} save steps; " \
    "found #{actual[0]} restore and #{actual[1]} save steps"
  )
end

counts.each do |workflow_name, actual|
  next if expected_counts.key?(workflow_name) || actual == [0, 0]

  errors << "#{workflow_name}: unexpected cache action; add it to the bounded policy deliberately"
end

unless errors.empty?
  warn("FAIL: GitHub Actions cache policy is not bounded:")
  errors.each { |error| warn("  - #{error}") }
  exit(1)
end

puts(
  "OK: repo-local Actions caches are source-only, epoch-bounded, and save only from main " \
  "(4 restores, 2 saves)."
)
RUBY
