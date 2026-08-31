#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workflow_root="${1:-${root}/.github/workflows}"

ruby - "${workflow_root}" <<'RUBY'
require "psych"
require "yaml"

workflow_root = File.expand_path(ARGV.fetch(0))
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

errors = []
counts = {}
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

  restore_count = 0
  save_count = 0

  jobs.each do |job_name, job|
    next unless job.is_a?(Hash)

    steps = job["steps"]
    next if steps.nil?
    unless steps.is_a?(Array)
      errors << "#{file_name}: job #{job_name.inspect} steps must be a YAML sequence"
      next
    end

    steps.each_with_index do |step, index|
      next unless step.is_a?(Hash)

      action = step["uses"]
      next unless action.is_a?(String) && action.start_with?("actions/cache")

      location = "#{file_name}: job #{job_name.inspect} step #{index + 1}"
      cache_paths = lines(step.dig("with", "path")) if step["with"].is_a?(Hash)
      key = step.dig("with", "key") if step["with"].is_a?(Hash)
      body = step.inspect

      case action
      when "actions/cache/restore@v6"
        restore_count += 1
        errors << "#{location}: restore id must be cargo-sources" unless step["id"] == "cargo-sources"
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
