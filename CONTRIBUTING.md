<!-- omit in toc -->
# Contributing to aisuite

First off, thanks for taking the time to contribute!

All types of contributions are encouraged and valued. See the [Table of Contents](#table-of-contents)
for different ways to help and details about how this project handles them. Please make sure to read
the relevant section before making your contribution. It will make it a lot easier for us maintainers
and smooth out the experience for all involved. The community looks forward to your contributions.

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy
> ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's readme
> - Mention the project at local meetups and tell your friends/colleagues

<!-- omit in toc -->
## Table of Contents

- [I Have a Question](#i-have-a-question)
- [I Want To Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving The Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)




## I Have a Question

> If you want to ask a question, we assume that you have read the available
> [Documentation](https://github.com/andrewyng/aisuite/blob/main/README.md).

Before you ask a question, it is best to search for existing [Issues](https://github.com/andrewyng/aisuite/issues)
that might help you. If you find a relevant issue that already exists and still need clarification, please add your question to that existing issue. We also recommend reaching out to the community in the aisuite [Discord](https://discord.gg/T6Nvn8ExSb) server.

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/andrewyng/aisuite/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (python, OS, etc.), depending on what seems relevant.

We (or someone in the community) will then take care of the issue as soon as possible.


## I Want To Contribute

> ### Legal Notice <!-- omit in toc -->
> When contributing to this project, you must agree that you have authored 100% of the content, that
> you have the necessary rights to the content and that the content you contribute may be provided
> under the project license.

### Reporting Bugs

<!-- omit in toc -->
#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask
you to investigate carefully, collect information and describe the issue in detail in your report. Please
complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side e.g. using incompatible environment 
  components/versions (Make sure that you have read the [documentation](https://github.com/andrewyng/aisuite/blob/main/README.md).
  If you are looking for support, you might want to check [this section](#i-have-a-question)).
- To see if other users have experienced (and potentially already solved) the same issue you are having,
  check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/andrewyng/aisuite?q=label%3Abug).
- Also make sure to search the internet (including Stack Overflow) to see if users outside of the GitHub
  community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of the interpreter, compiler, SDK, runtime environment, package manager, depending on
    what seems relevant.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

<!-- omit in toc -->
#### How Do I Submit a Good Bug Report?

> You must never report security related issues, vulnerabilities or bugs including sensitive information to
> the issue tracker, or elsewhere in public. Instead sensitive bugs must be sent by email to <joaquin.dominguez@proton.me>.
<!-- You may add a PGP key to allow the messages to be sent encrypted as well. -->

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/andrewyng/aisuite/issues/new). (Since we can't be sure at
  this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can
  follow to recreate the issue on their own. This usually includes your code. For good bug reports you
  should isolate the problem and create a reduced test case.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction 
  steps or no obvious way to reproduce the issue, the team will ask you for those steps and mark the
  issue as `needs-repro`. Bugs with the `needs-repro` tag will not be addressed until they are reproduced.
- If the team is able to reproduce the issue, it will be marked `needs-fix`, as well as possibly other
  tags (such as `critical`), and the issue will be left to be
  [implemented by someone](#your-first-code-contribution).

Please use the issue templates provided.


### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for aisuite,
**including completely new features and minor improvements to existing functionality**. Following these
guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

<!-- omit in toc -->
#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://github.com/andrewyng/aisuite/blob/main/README.md) carefully
  and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/andrewyng/aisuite/issues) to see if the enhancement has
  already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong
  case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

<!-- omit in toc -->
#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/andrewyng/aisuite/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
  At this point you can also tell which alternatives do not work for you.
- **Explain why this enhancement would be useful** to most aisuite users. You may also want to
  point out the other projects that solved it better and which could serve as inspiration.


### Your First Code Contribution

#### Pre-requisites

You should first [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
the `aisuite` repository and then clone your forked repository:

```bash
git clone https://github.com/<YOUR_GITHUB_USER>/aisuite.git
```



Once in the cloned repository directory, make a branch on the forked repository with your username and
description of PR:
```bash
git checkout -B <username>/<description>
```

Please install the development and test dependencies:
```bash
poetry install --with dev,test
```

`aisuite` uses pre-commit to ensure the formatting is consistent:
```bash
pre-commit install
```

**Make suggested changes**

Afterwards, our suite of formatting tests will run automatically before each `git commit`. You can also
run these manually:
```bash
pre-commit run --all-files
```

If a formatting test fails, it will fix the modified code in place and abort the `git commit`. After looking
over the changes, you can `git add <modified files>` and then repeat the previous git commit command.

**Note**: a github workflow will check the files with the same formatter and reject the PR if it doesn't
pass, so please make sure it passes locally.


#### Testing
`aisuite` tracks unit tests. Pytest is used to execute said unit tests in `tests/`:

```bash
poetry run pytest tests
```

If your code changes implement a new function, please make a corresponding unit test to the `test/*` files.

#### Contributing Workflow
We actively welcome your pull requests.

1. Create your new branch from main in your forked repo, with your username and a name describing the work
   you're completing e.g. user-123/add-feature-x.
2. If you've added code that should be tested, add tests. Ensure all tests pass. See the testing section
   for more information.
3. If you've changed APIs, update the documentation.
4. Make sure your code lints.



### Improving The Documentation
We welcome valuable contributions in the form of new documentation or revised documentation that provide
further clarity or accuracy. Each function should be clearly documented. Well-documented code is easier
to review and understand/extend.

## Styleguides
For code documentation, please follow the [Google styleguide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).
