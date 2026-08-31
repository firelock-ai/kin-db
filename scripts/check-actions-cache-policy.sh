#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workflow_root="${1:-${root}/.github/workflows}"
action_root="${2:-$(dirname "${workflow_root}")/actions}"
repo_root="${3:-$(cd "${workflow_root}/../.." && pwd)}"

ruby - "${workflow_root}" "${action_root}" "${repo_root}" "${root}" <<'RUBY'
require "digest"
require "find"
require "json"
require "psych"
require "set"
require "yaml"

workflow_root = File.expand_path(ARGV.fetch(0))
action_root = File.expand_path(ARGV.fetch(1))
repo_root = File.expand_path(ARGV.fetch(2))
authority_root = File.expand_path(ARGV.fetch(3))

CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
CACHE_RESTORE_ACTION = "actions/cache/restore@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"
CACHE_SAVE_ACTION = "actions/cache/save@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"
UNSAFE_REGISTRY_COMMIT = "d6b6585d0b5902437d2745a94a960fe0d7d27f0e"

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
  "EmbarkStudios/cargo-deny-action@3c6349835b2b7b196a839186cb8b78e02f7b5f25",
  CACHE_RESTORE_ACTION,
  CACHE_SAVE_ACTION,
  CHECKOUT_ACTION,
  "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
  "codecov/codecov-action@0fb7174895f61a3b6b78fc075e0cd60383518dac",
  "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65",
  "taiki-e/install-action@1ed6d7be6168f6c9046541087ff549b6bc581fdf",
]).freeze
expected_action_identities = {
  ["cache-policy-authority.yml", "candidate-policy", 0] => CHECKOUT_ACTION,
  ["cache-policy-authority.yml", "candidate-policy", 1] => CHECKOUT_ACTION,
  ["ci-linux.yml", "linux-build-test", 0] => CHECKOUT_ACTION,
  ["ci-linux.yml", "linux-build-test", 1] => "./kin-db/.github/actions/rust-toolchain",
  ["ci-linux.yml", "linux-build-test", 2] => CACHE_RESTORE_ACTION,
  ["ci.yml", "dco", 0] => CHECKOUT_ACTION,
  ["ci.yml", "check", 0] => CHECKOUT_ACTION,
  ["ci.yml", "check", 1] => "./.github/actions/rust-toolchain",
  ["ci.yml", "check", 2] => CACHE_RESTORE_ACTION,
  ["ci.yml", "check", 10] => CACHE_SAVE_ACTION,
  ["ci.yml", "schema-provenance", 0] => CHECKOUT_ACTION,
  ["ci.yml", "coverage", 0] => CHECKOUT_ACTION,
  ["ci.yml", "coverage", 1] => "./.github/actions/rust-toolchain",
  ["ci.yml", "coverage", 2] => CACHE_RESTORE_ACTION,
  ["ci.yml", "coverage", 3] => "taiki-e/install-action@1ed6d7be6168f6c9046541087ff549b6bc581fdf",
  ["ci.yml", "coverage", 5] => "codecov/codecov-action@0fb7174895f61a3b6b78fc075e0cd60383518dac",
  ["fuzz.yml", "fuzz-smoke", 0] => CHECKOUT_ACTION,
  ["fuzz.yml", "fuzz-smoke", 1] => "./.github/actions/rust-toolchain",
  ["fuzz.yml", "fuzz-smoke", 4] => "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
  ["release.yml", "validate", 0] => CHECKOUT_ACTION,
  ["release.yml", "validate", 1] => "./.github/actions/rust-toolchain",
  ["release.yml", "release", 0] => CHECKOUT_ACTION,
  ["release.yml", "release", 1] => "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65",
  ["sast.yml", "cargo-deny", 0] => CHECKOUT_ACTION,
  ["sast.yml", "cargo-deny", 1] => "./.github/actions/rust-toolchain",
  ["sast.yml", "cargo-deny", 2] => "EmbarkStudios/cargo-deny-action@3c6349835b2b7b196a839186cb8b78e02f7b5f25",
  ["windows-nightly.yml", "windows", 0] => CHECKOUT_ACTION,
  ["windows-nightly.yml", "windows", 1] => "./.github/actions/rust-toolchain",
  ["windows-nightly.yml", "windows", 2] => CACHE_RESTORE_ACTION,
  ["windows-nightly.yml", "windows", 10] => CACHE_SAVE_ACTION,
}.freeze
approved_reusable_jobs = {
  ["kin-dependency-wave.yml", "dependency-wave"] => {
    uses: "firelock-ai/kin-actions/.github/workflows/cargo-dependency-wave.yml@d6b6585d0b5902437d2745a94a960fe0d7d27f0e",
    digest: "06602f164d9a5eacdc3aaade5325bad5e637b62f7eb820915e36a581ee394f6c",
  },
  ["scheduled-failure-alarm.yml", "alarm"] => {
    uses: "firelock-ai/kin-actions/.github/workflows/scheduled-failure-alarm.yml@398595fa14ba1eaebca6eb176facd8a57ce9db05",
    digest: "358f14736aa565ee219db8c47f68043117ee54760bde937ed813b78b1e77bc7d",
  },
}.freeze
protected_job_contracts = {
  ["ci.yml", "check"] => [
    "54440a78bb977b5ec0c72cf26ba2d5886ff777c2f0da81033bbb9b7a8b43c953",
    "cache-owner job contract drifted",
  ],
  ["ci.yml", "schema-provenance"] => [
    "0a561846cdd51c4d0f4203454b0a3cc42d36e0aa0186b791a509870169ec321b",
    "candidate self-check job drifted or became non-authoritative",
  ],
  ["ci.yml", "coverage"] => [
    "06924d5f750ef317e9c1b1cea5a95cd071572d9dc6a9c2dfba5494ae0b4ca4e1",
    "coverage cache-owner job contract drifted",
  ],
  ["ci-linux.yml", "linux-build-test"] => [
    "fc4353c3f707b83fd6fa4bfb8426f36d1e5908c652d93d773d4c78c0e2a752e5",
    "Linux cache-owner job contract drifted",
  ],
  ["windows-nightly.yml", "windows"] => [
    "f767a34ff9454ad28523997c61c228b85fad15a6909dd14762e7735789e4019b",
    "Windows cache-owner job contract drifted",
  ],
  ["registry-publish.yml", "release"] => [
    "37191df75202f9f5dd669fe74734aa151420b38cbc92ebcbd669c844327c90dd",
    "registry release hold drifted or became non-authoritative",
  ],
  ["cache-policy-authority.yml", "candidate-policy"] => [
    "9f72550e78de36b536a3bbfc30f68d5f20b6197fe022744fcbcebb058208d487",
    "trusted authority job drifted or became non-authoritative",
  ],
}.freeze
protected_workflow_envelopes = {
  "cache-policy-authority.yml" => "e46706ae86086649d68c2832e3da6643d28ac3e5199e064e3b20a33dbdb4a9b8",
  "ci.yml" => "a38b83967e34cdbced7990a548c240f68bf728e304f43d5828b93130b13243ae",
  "ci-linux.yml" => "9474a4b88c83c3eb3f66682b2ff14ac16debae0d9b819ec71ff49088226edd74",
  "windows-nightly.yml" => "d62f3a6e796150c7bf2cf9e22c22aeba509acc52821c413eadb6117447c7cec3",
  "registry-publish.yml" => "21dd138161e756f2df3409237e8b2cb341146c5f0b11b8818eab3116cdd8b3dc",
}.freeze
protected_repo_files = {
  ".github/actions/rust-toolchain/action.yml" => [
    "6f33e96a8bfd31511b907d0b22a54be94ccfde79bf2b328f7aa56e087de8d398",
    "protected local rust-toolchain action drifted",
  ],
  ".cargo/config.toml" => [
    "9bd43ae87aad7adc817737b91249b1c44c92b415a43757a88128dab8ab63c8e6",
    "tracked Cargo config drifted or can redirect target output",
  ],
  "rust-toolchain.toml" => [
    "796c340f29e77e94c0e4c6d6d6cac0ae3f07f88ad68622f338116a4a489dd756",
    "Rust toolchain pin drifted",
  ],
  ".github/workflows/cache-policy-authority.yml" => [
    "cb87ccd555cc8cb39e9dc5547d03d8278afeed92d9f6618def6a338e0dd5e232",
    "trusted authority workflow drifted or became non-authoritative",
  ],
}.freeze
trusted_policy_files = %w[
  .github/workflows/cache-policy-authority.yml
  scripts/check-actions-cache-policy.sh
  scripts/test-actions-cache-policy.sh
].freeze
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
    "@#{CHECKOUT_ACTION}",
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
    "@#{CHECKOUT_ACTION}",
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
protected_jobs_seen = Set.new
action_identities_seen = Set.new

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

def canonical(value)
  case value
  when Hash
    value.keys.sort_by(&:to_s).to_h { |key| [key.to_s, canonical(value.fetch(key))] }
  when Array
    value.map { |entry| canonical(entry) }
  else
    value
  end
end

def canonical_digest(value)
  Digest::SHA256.hexdigest(JSON.generate(canonical(value)))
end

def path_within?(path, root)
  resolved_path = File.realpath(path)
  resolved_root = File.realpath(root)
  resolved_path == resolved_root || resolved_path.start_with?("#{resolved_root}#{File::SEPARATOR}")
rescue SystemCallError
  false
end

def regular_file?(path, root, location, errors)
  stat = File.lstat(path)
  unless stat.file? && !stat.symlink?
    errors << "#{location}: must be a regular file, not a symlink or special file"
    return false
  end
  unless path_within?(path, root)
    errors << "#{location}: resolves outside the candidate repository"
    return false
  end

  true
rescue Errno::ENOENT, Errno::ENOTDIR
  errors << "#{location}: required file is missing"
  false
end

def safe_directory?(path, root, location, errors)
  stat = File.lstat(path)
  unless stat.directory? && !stat.symlink?
    errors << "#{location}: must be a real directory, not a symlink"
    return false
  end
  unless path_within?(path, root)
    errors << "#{location}: resolves outside the candidate repository"
    return false
  end

  true
rescue Errno::ENOENT, Errno::ENOTDIR
  errors << "#{location}: required directory is missing"
  false
end

def immutable_remote_reference?(action)
  action.is_a?(String) && action.match?(%r{\A[^@\s]+@[0-9a-f]{40}\z})
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

def inspect_github_env_writes(mapping, location, errors)
  run = mapping["run"]
  return unless run.is_a?(String) && run.match?(/GITHUB_ENV/i)
  return unless run.match?(/CARGO_TARGET_DIR|(?:^|[^A-Z_])HOME\s*=/i)

  errors << "#{location}: protected target or HOME write through GITHUB_ENV is forbidden"
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

unless safe_directory?(repo_root, repo_root, "candidate repository", errors)
  warn("FAIL: GitHub Actions cache policy could not inspect the candidate repository:")
  errors.each { |error| warn("  - #{error}") }
  exit(1)
end
workflow_root_ok = safe_directory?(workflow_root, repo_root, "workflow root", errors)
action_root_ok = safe_directory?(action_root, repo_root, "local action root", errors)
authority_root_ok = safe_directory?(authority_root, authority_root, "trusted authority root", errors)

trusted_policy_files.each do |relative_path|
  candidate_path = File.join(repo_root, relative_path)
  trusted_path = File.join(authority_root, relative_path)
  candidate_ok = regular_file?(candidate_path, repo_root, relative_path, errors)
  trusted_ok = authority_root_ok && regular_file?(trusted_path, authority_root, "trusted #{relative_path}", errors)
  next unless candidate_ok && trusted_ok

  candidate_digest = Digest::SHA256.file(candidate_path).hexdigest
  trusted_digest = Digest::SHA256.file(trusted_path).hexdigest
  candidate_mode = File.stat(candidate_path).mode & 0o111
  trusted_mode = File.stat(trusted_path).mode & 0o111
  unless candidate_digest == trusted_digest && candidate_mode == trusted_mode
    errors << "#{relative_path}: trusted policy implementation drifted from default-branch authority"
  end
end

protected_repo_files.each do |relative_path, (expected_digest, message)|
  path = File.join(repo_root, relative_path)
  next unless regular_file?(path, repo_root, relative_path, errors)

  errors << "#{relative_path}: #{message}" unless Digest::SHA256.file(path).hexdigest == expected_digest
end

legacy_cargo_config = File.join(repo_root, ".cargo/config")
if File.exist?(legacy_cargo_config) || File.symlink?(legacy_cargo_config)
  errors << ".cargo/config: legacy tracked Cargo config is forbidden; it can redirect target output"
end

workflows = workflow_root_ok ? Dir[File.join(workflow_root, "*.{yml,yaml}")].sort : []
if workflows.empty?
  errors << "no workflow files found under #{workflow_root}"
end

workflows.each do |workflow|
  file_name = File.basename(workflow)
  next unless regular_file?(workflow, repo_root, file_name, errors)

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

  if protected_workflow_envelopes.key?(file_name)
    envelope = document.reject { |key, _value| key == "jobs" }
    unless canonical_digest(envelope) == protected_workflow_envelopes.fetch(file_name)
      errors << "#{file_name}: protected workflow trigger, permission, or global contract drifted"
    end
  end

  if document["name"] == "Cache Policy Authority" && file_name != "cache-policy-authority.yml"
    errors << "#{file_name}: duplicate Cache Policy Authority workflow identity is forbidden"
  end

  if document.key?("defaults")
    errors << "#{file_name}: workflow defaults are forbidden because they can mask guard failures"
  end

  each_mapping(document) do |mapping|
    inspect_protected_env(mapping, file_name, protected_env_keys, errors)
    inspect_github_env_writes(mapping, file_name, errors)
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
               checkout["uses"] == CHECKOUT_ACTION
          errors << "ci.yml: schema-provenance must begin with the exact immutable checkout identity"
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

    identity = [file_name, job_name]
    if job["name"] == "Cache Policy Authority" && identity != ["cache-policy-authority.yml", "candidate-policy"]
      errors << "#{file_name}: job #{job_name.inspect} duplicates the trusted Cache Policy Authority identity"
    end

    if protected_job_contracts.key?(identity)
      expected_digest, message = protected_job_contracts.fetch(identity)
      protected_jobs_seen << identity
      unless canonical_digest(job) == expected_digest
        errors << "#{file_name}: job #{job_name.inspect} #{message}"
      end
    end

    reusable = job["uses"]
    if reusable
      if reusable.to_s.include?("cargo-registry-release.yml@") &&
         (reusable.to_s.end_with?("@v0.1.31") || reusable.to_s.end_with?("@#{UNSAFE_REGISTRY_COMMIT}"))
        errors << (
          "#{file_name}: job #{job_name.inspect} unsafe target-caching registry release is forbidden " \
          "until reviewed kin-actions v0.1.34 immutable bytes exist"
        )
      end
      unless immutable_remote_reference?(reusable)
        errors << "#{file_name}: job #{job_name.inspect} reusable workflow must use a full immutable commit SHA"
      end
      expected = approved_reusable_jobs[identity]
      if !expected || expected.fetch(:uses) != reusable
        errors << "#{file_name}: job #{job_name.inspect} uses unapproved reusable workflow #{reusable.inspect}"
      elsif canonical_digest(job) != expected.fetch(:digest)
        errors << "#{file_name}: job #{job_name.inspect} reviewed reusable-job mapping drifted"
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
      unless checkout.is_a?(Hash) && checkout.keys == ["uses"] && checkout["uses"] == CHECKOUT_ACTION
        errors << "#{file_name}: cache-owner job must begin with the exact immutable checkout identity"
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
      action_identity = [file_name, job_name, index]
      if action
        expected_action = expected_action_identities[action_identity]
        if expected_action
          action_identities_seen << action_identity
          unless action == expected_action
            errors << "#{location}: action identity drifted from #{expected_action.inspect}"
          end
        else
          errors << "#{location}: unexpected action-bearing step is not part of reviewed topology"
        end
      end
      if action.is_a?(String) && !action.start_with?("./") && !immutable_remote_reference?(action)
        errors << "#{location}: remote action must use a full immutable commit SHA"
      end
      if action && (!action.is_a?(String) || !approved_step_actions.include?(action))
        errors << "#{location}: unapproved action identity #{action.inspect}"
      end
      next unless action.is_a?(String) && action.downcase.start_with?("actions/cache")

      cache_paths = lines(step.dig("with", "path")) if step["with"].is_a?(Hash)
      key = step.dig("with", "key") if step["with"].is_a?(Hash)
      body = step.inspect

      case action
      when CACHE_RESTORE_ACTION
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
      when CACHE_SAVE_ACTION
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
                   prior["uses"] == CACHE_RESTORE_ACTION
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

protected_job_contracts.each_key do |identity|
  unless protected_jobs_seen.include?(identity)
    errors << "#{identity[0]}: required protected job #{identity[1].inspect} is missing"
  end
end

expected_action_identities.each_key do |identity|
  unless action_identities_seen.include?(identity)
    errors << "#{identity[0]}: required reviewed action at job #{identity[1].inspect} step #{identity[2] + 1} is missing"
  end
end

approved_reusable_jobs.each_key do |identity|
  unless reusable_seen.include?(identity)
    errors << "#{identity[0]}: required reviewed reusable job #{identity[1].inspect} is missing"
  end
end

action_files = []
if action_root_ok
  Find.find(action_root) do |path|
    stat = File.lstat(path)
    if stat.symlink?
      relative_name = path.delete_prefix("#{action_root}/")
      errors << "#{relative_name}: symlinks are forbidden under the local action root"
      Find.prune if File.directory?(path)
    elsif stat.file? && path.match?(/\.ya?ml\z/)
      action_files << path
    end
  rescue Errno::ENOENT, Errno::ENOTDIR
    errors << "#{path}: disappeared during local action inspection"
  end
end

action_files.sort.each do |action_file|
  relative_name = action_file.delete_prefix("#{action_root}/")
  next unless regular_file?(action_file, repo_root, relative_name, errors)

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
    inspect_github_env_writes(mapping, relative_name, errors)
    inspect_hidden_cache_action(mapping, relative_name, errors)
    action = mapping["uses"]
    if action.is_a?(String) && !action.start_with?("./") && !immutable_remote_reference?(action)
      errors << "#{relative_name}: remote composite action must use a full immutable commit SHA"
    end
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
