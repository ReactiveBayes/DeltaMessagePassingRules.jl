# The documentation site of DeltaMessagePassingRules. Build it with `make docs`, or
# `julia --project=docs docs/make.jl`.
using Documenter, DocumenterInterLinks, DocInventories
using DeltaMessagePassingRules

# A registered package's site, by its published inventory. A first request to GitHub Pages can take
# seconds, longer than DocInventories' default timeout of one second, so it waits longer.
registered(name) = Inventory(
    "https://reactivebayes.github.io/$(name).jl/stable/objects.inv";
    root_url = "https://reactivebayes.github.io/$(name).jl/stable/", timeout = 30,
)

links = InterLinks(
    "MessagePassingRulesApproximations" => registered("MessagePassingRulesApproximations"),
    "MessagePassingRulesBase" => registered("MessagePassingRulesBase"),
)

DocMeta.setdocmeta!(
    DeltaMessagePassingRules,
    :DocTestSetup,
    :(using DeltaMessagePassingRules, MessagePassingRulesBase, MessagePassingRulesApproximations, ExponentialFamily, BayesBase);
    recursive = true,
)

makedocs(
    modules = [DeltaMessagePassingRules],
    sitename = "DeltaMessagePassingRules.jl",
    # Every docstring appears on a page.
    checkdocs = :all,
    plugins = [links],
    pages = ["Overview" => "index.md", "The Delta node" => "delta.md", "Projection" => "projection.md", "Internals" => "internals.md"],
    # Pages that show rule results as cards are larger than the defaults allow.
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        example_size_threshold = 400 * 1024, size_threshold_warn = 400 * 1024, size_threshold = 400 * 1024,
    ),
)

deploydocs(
    repo = "github.com/ReactiveBayes/DeltaMessagePassingRules.jl.git",
    devbranch = "main",
    # A preview for every pull request but CompatHelper's, whose documentation nobody reviews;
    # `DocPreviewCleanup.yml` deletes it when the pull request closes.
    push_preview = !startswith(get(ENV, "GITHUB_HEAD_REF", ""), "compathelper/"),
)
