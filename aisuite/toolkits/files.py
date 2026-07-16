from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..agents import ToolMetadata, tool

DEFAULT_IGNORES = (
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
)


def files(
    *,
    root: str | Path | None = None,
    roots: Optional[list] = None,
    allow_write: bool = False,
    max_read_bytes: int = 200_000,
    max_search_bytes: int = 1_000_000,
    ignore: Optional[list[str]] = None,
) -> list:
    """Return filesystem tools scoped to one or more roots.

    Single-root (back-compat): pass `root=<path>` with `allow_write`. Multi-root: pass
    `roots=[{"path": ..., "writable": bool}, ...]` (or RootDir-like objects / a live mutable
    list, read on every call so runtime add/remove takes effect). Index 0 is the primary:
    relative paths resolve against it and tool results display relative to it. Reads are
    allowed under any root; writes only under a writable one. Write tools are exposed when
    `allow_write` OR any root is writable.
    """
    toolkit = FileToolkit(
        root=root,
        roots=roots,
        allow_write=allow_write,
        max_read_bytes=max_read_bytes,
        max_search_bytes=max_search_bytes,
        ignore=ignore,
    )
    allow_write = toolkit.has_writable

    def list_files(
        path: str = ".",
        pattern: str = "*",
        recursive: bool = True,
        max_results: int = 100,
    ) -> list[str]:
        """List files under the configured root."""
        return toolkit.list_files(
            path=path,
            pattern=pattern,
            recursive=recursive,
            max_results=max_results,
        )

    def read_file(path: str) -> str:
        """Read a UTF-8 text file under the configured root."""
        return toolkit.read_file(path=path)

    def read_file_lines(
        path: str,
        start_line: int = 1,
        max_lines: int = 100,
    ) -> dict[str, object]:
        """Read a line range from a UTF-8 text file under the configured root."""
        return toolkit.read_file_lines(
            path=path,
            start_line=start_line,
            max_lines=max_lines,
        )

    def search_files(
        query: str,
        path: str = ".",
        pattern: str = "*",
        max_results: int = 50,
    ) -> list[dict[str, object]]:
        """Search UTF-8 text files under the configured root."""
        return toolkit.search_files(
            query=query,
            path=path,
            pattern=pattern,
            max_results=max_results,
        )

    tools = [
        tool(
            list_files,
            metadata=ToolMetadata(
                category="filesystem",
                risk_level="low",
                capabilities=["list_files"],
            ),
        ),
        tool(
            read_file,
            metadata=ToolMetadata(
                category="filesystem",
                risk_level="low",
                capabilities=["read_file"],
            ),
        ),
        tool(
            read_file_lines,
            metadata=ToolMetadata(
                category="filesystem",
                risk_level="low",
                capabilities=["read_file_lines"],
            ),
        ),
        tool(
            search_files,
            metadata=ToolMetadata(
                category="filesystem",
                risk_level="low",
                capabilities=["search_files"],
            ),
        ),
    ]
    if allow_write:

        def write_file(path: str, content: str, overwrite: bool = True) -> str:
            """Write a UTF-8 text file under the configured root."""
            return toolkit.write_file(
                path=path,
                content=content,
                overwrite=overwrite,
            )

        def apply_unified_diff(diff: str) -> dict[str, object]:
            """Apply a unified diff patch under the configured root."""
            return toolkit.apply_unified_diff(diff=diff)

        def apply_patch(patch: str) -> dict[str, object]:
            """Apply a Codex-style patch envelope under the configured root."""
            return toolkit.apply_patch(patch=patch)

        def replace_in_file(
            path: str,
            old: str,
            new: str,
            expected_replacements: int = 1,
        ) -> dict[str, object]:
            """Replace an exact text fragment in a UTF-8 file under the configured root."""
            return toolkit.replace_in_file(
                path=path,
                old=old,
                new=new,
                expected_replacements=expected_replacements,
            )

        tools.append(
            tool(
                write_file,
                metadata=ToolMetadata(
                    category="filesystem",
                    risk_level="medium",
                    capabilities=["write_file"],
                    requires_approval=True,
                ),
            )
        )
        tools.append(
            tool(
                apply_unified_diff,
                metadata=ToolMetadata(
                    category="filesystem",
                    risk_level="medium",
                    capabilities=["apply_patch"],
                    requires_approval=True,
                ),
            )
        )
        tools.append(
            tool(
                apply_patch,
                metadata=ToolMetadata(
                    category="filesystem",
                    risk_level="medium",
                    capabilities=["apply_patch"],
                    requires_approval=True,
                    description=(
                        "Apply a Codex-style patch using *** Begin Patch / "
                        "*** End Patch sections."
                    ),
                ),
            )
        )
        tools.append(
            tool(
                replace_in_file,
                metadata=ToolMetadata(
                    category="filesystem",
                    risk_level="medium",
                    capabilities=["edit_file"],
                    requires_approval=True,
                    description="Replace an exact text fragment in one file.",
                ),
            )
        )
    return tools


@dataclass
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class _FilePatch:
    old_path: str
    new_path: str
    hunks: list[_Hunk]


class FileToolkit:
    def __init__(
        self,
        *,
        root: str | Path | None = None,
        roots: Optional[list] = None,
        allow_write: bool = False,
        max_read_bytes: int,
        max_search_bytes: int,
        ignore: Optional[list[str]],
    ):
        # `roots` is kept by reference and re-read on every resolution, so a caller that mutates
        # the shared list (add/remove a folder mid-session) takes effect without rebuilding tools.
        if roots is None:
            if root is None:
                raise ValueError("files() requires `root` or `roots`")
            roots = [{"path": root, "writable": allow_write}]
        self._roots_ref = roots
        self.allow_write = allow_write
        self.max_read_bytes = max_read_bytes
        self.max_search_bytes = max_search_bytes
        self.ignore = set(DEFAULT_IGNORES if ignore is None else ignore)
        norm = self._roots()
        if not norm:
            raise ValueError("files() requires at least one root")
        self.root = norm[0][
            0
        ]  # primary: stable, used for relative resolution + display
        self.has_writable = any(writable for _, writable in norm)
        self.root.mkdir(parents=True, exist_ok=True)

    def _roots(self) -> list[tuple[Path, bool]]:
        """Live snapshot of (resolved path, writable) for each configured root."""
        out: list[tuple[Path, bool]] = []
        for r in self._roots_ref:
            if isinstance(r, dict):
                p, w = r["path"], bool(r.get("writable", False))
            elif isinstance(r, (str, Path)):
                p, w = r, self.allow_write
            else:  # duck-typed RootDir-like
                p, w = getattr(r, "path"), bool(getattr(r, "writable", False))
            out.append((Path(p).expanduser().resolve(), w))
        return out

    def _root_for(self, candidate: Path) -> Optional[tuple[Path, bool]]:
        for rp, writable in self._roots():
            try:
                candidate.relative_to(rp)
                return (rp, writable)
            except ValueError:
                continue
        return None

    def list_files(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = True,
        max_results: int = 100,
    ) -> list[str]:
        """List files under the configured root."""
        base = self._resolve(path)
        if not base.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not base.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        iterator = base.rglob(pattern) if recursive else base.glob(pattern)
        results = []
        for item in iterator:
            if self._ignored(item):
                continue
            if item.is_file():
                results.append(self._relative(item))
            if len(results) >= max_results:
                break
        return sorted(results)

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file under the configured root."""
        file_path = self._resolve_file(path)
        size = file_path.stat().st_size
        if size > self.max_read_bytes:
            raise ValueError(
                f"File exceeds max_read_bytes ({self.max_read_bytes}): {path}"
            )
        return file_path.read_text(encoding="utf-8")

    def read_file_lines(
        self,
        path: str,
        start_line: int = 1,
        max_lines: int = 100,
    ) -> dict[str, object]:
        """Read a line range from a UTF-8 text file under the configured root."""
        if start_line < 1:
            raise ValueError("start_line must be >= 1")
        if max_lines < 1:
            raise ValueError("max_lines must be >= 1")

        content = self.read_file(path)
        lines = content.splitlines()
        start_index = start_line - 1
        selected = lines[start_index : start_index + max_lines]
        end_line = start_line + len(selected) - 1 if selected else start_line - 1
        return {
            "path": self._relative(self._resolve_file(path)),
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": len(lines),
            "content": "\n".join(selected),
        }

    def search_files(
        self,
        query: str,
        path: str = ".",
        pattern: str = "*",
        max_results: int = 50,
    ) -> list[dict[str, object]]:
        """Search UTF-8 text files under the configured root."""
        base = self._resolve(path)
        if not base.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not base.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        results = []
        scanned_bytes = 0
        for item in base.rglob(pattern):
            if self._ignored(item):
                continue
            if not item.is_file():
                continue
            size = item.stat().st_size
            if scanned_bytes + size > self.max_search_bytes:
                break
            scanned_bytes += size
            matches = self._search_file(item, query)
            results.extend(matches)
            if len(results) >= max_results:
                return results[:max_results]
        return results

    def write_file(self, path: str, content: str, overwrite: bool = True) -> str:
        """Write a UTF-8 text file under the configured root."""
        if not self.has_writable:
            raise PermissionError("write_file is disabled for this file toolkit.")
        file_path = self._resolve(path, for_write=True)
        if file_path.exists() and file_path.is_dir():
            raise ValueError(f"Path is a directory: {path}")
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return self._relative(file_path)

    def apply_unified_diff(self, diff: str) -> dict[str, object]:
        """Apply a unified diff patch under the configured root."""
        if not self.has_writable:
            raise PermissionError(
                "apply_unified_diff is disabled for this file toolkit."
            )

        patches = self._parse_unified_diff(diff)
        changed_files = []
        added_files = []
        deleted_files = []
        hunk_count = 0

        for patch in patches:
            old_is_dev_null = patch.old_path == "/dev/null"
            new_is_dev_null = patch.new_path == "/dev/null"
            target_path = patch.old_path if new_is_dev_null else patch.new_path
            resolved = self._resolve_diff_path(target_path, for_write=True)
            old_lines = [] if old_is_dev_null else self._read_patch_lines(resolved)
            new_lines = self._apply_hunks(old_lines, patch.hunks, target_path)
            hunk_count += len(patch.hunks)

            if new_is_dev_null:
                if resolved.exists():
                    if not resolved.is_file():
                        raise ValueError(f"Path is not a file: {target_path}")
                    resolved.unlink()
                deleted_files.append(self._relative(resolved))
                continue

            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text("".join(new_lines), encoding="utf-8")
            relative = self._relative(resolved)
            changed_files.append(relative)
            if old_is_dev_null:
                added_files.append(relative)

        return {
            "changed_files": changed_files,
            "added_files": added_files,
            "deleted_files": deleted_files,
            "file_count": len(patches),
            "hunk_count": hunk_count,
        }

    def replace_in_file(
        self,
        path: str,
        old: str,
        new: str,
        expected_replacements: int = 1,
    ) -> dict[str, object]:
        """Replace an exact text fragment in a UTF-8 file under the configured root."""
        if not self.has_writable:
            raise PermissionError("replace_in_file is disabled for this file toolkit.")
        if not old:
            raise ValueError("old must be a non-empty string.")
        if expected_replacements < 1:
            raise ValueError("expected_replacements must be >= 1")
        file_path = self._resolve_file(path, for_write=True)
        content = file_path.read_text(encoding="utf-8")
        count = content.count(old)
        if count != expected_replacements:
            raise ValueError(
                f"Expected {expected_replacements} replacement(s), found {count}: {path}"
            )
        updated = content.replace(old, new, expected_replacements)
        file_path.write_text(updated, encoding="utf-8")
        return {
            "path": self._relative(file_path),
            "replacements": count,
            "chars_before": len(content),
            "chars_after": len(updated),
        }

    def apply_patch(self, patch: str) -> dict[str, object]:
        """Apply a Codex-style patch envelope under the configured root."""
        if not self.has_writable:
            raise PermissionError("apply_patch is disabled for this file toolkit.")

        lines = patch.splitlines(keepends=True)
        if not lines or lines[0].strip() != "*** Begin Patch":
            raise ValueError("Patch must start with *** Begin Patch.")
        if lines[-1].strip() != "*** End Patch":
            raise ValueError("Patch must end with *** End Patch.")

        changed_files: list[str] = []
        added_files: list[str] = []
        deleted_files: list[str] = []
        file_count = 0
        hunk_count = 0
        index = 1

        while index < len(lines) - 1:
            line = lines[index].rstrip("\r\n")
            if not line:
                index += 1
                continue
            if line.startswith("*** Add File: "):
                path = line[len("*** Add File: ") :].strip()
                index += 1
                content_lines = []
                while index < len(lines) - 1 and not lines[index].startswith("*** "):
                    current = lines[index]
                    if not current.startswith("+"):
                        raise ValueError(f"Invalid add-file patch line: {current}")
                    content_lines.append(current[1:])
                    index += 1
                resolved = self._resolve(path, for_write=True)
                if resolved.exists():
                    raise FileExistsError(f"File already exists: {path}")
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text("".join(content_lines), encoding="utf-8")
                relative = self._relative(resolved)
                changed_files.append(relative)
                added_files.append(relative)
                file_count += 1
                hunk_count += 1
                continue

            if line.startswith("*** Delete File: "):
                path = line[len("*** Delete File: ") :].strip()
                resolved = self._resolve_file(path, for_write=True)
                resolved.unlink()
                relative = self._relative(resolved)
                deleted_files.append(relative)
                file_count += 1
                hunk_count += 1
                index += 1
                continue

            if line.startswith("*** Update File: "):
                path = line[len("*** Update File: ") :].strip()
                resolved = self._resolve_file(path, for_write=True)
                index += 1
                move_to = None
                if index < len(lines) - 1 and lines[index].startswith("*** Move to: "):
                    move_to = lines[index][len("*** Move to: ") :].strip()
                    index += 1
                chunk_lines: list[list[str]] = []
                current_chunk: list[str] = []
                while index < len(lines) - 1 and not lines[index].startswith("*** "):
                    current = lines[index]
                    if current.startswith("@@"):
                        if current_chunk:
                            chunk_lines.append(current_chunk)
                            current_chunk = []
                    elif current and current[0] in {" ", "+", "-"}:
                        current_chunk.append(current)
                    else:
                        raise ValueError(f"Invalid update patch line: {current}")
                    index += 1
                if current_chunk:
                    chunk_lines.append(current_chunk)
                if not chunk_lines:
                    raise ValueError(f"Patch update has no hunks for {path}.")

                old_lines = resolved.read_text(encoding="utf-8").splitlines(
                    keepends=True
                )
                new_lines, applied_hunks = self._apply_patch_chunks(
                    old_lines,
                    chunk_lines,
                    path,
                )
                target = self._resolve(move_to, for_write=True) if move_to else resolved
                if move_to and target.exists() and target != resolved:
                    raise FileExistsError(f"File already exists: {move_to}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("".join(new_lines), encoding="utf-8")
                if move_to and target != resolved:
                    resolved.unlink()
                    deleted_files.append(self._relative(resolved))
                changed_files.append(self._relative(target))
                file_count += 1
                hunk_count += applied_hunks
                continue

            raise ValueError(f"Unknown patch operation: {line}")

        if file_count == 0:
            raise ValueError("Patch contains no file operations.")
        return {
            "changed_files": changed_files,
            "added_files": added_files,
            "deleted_files": deleted_files,
            "file_count": file_count,
            "hunk_count": hunk_count,
        }

    def _resolve(self, path: str, *, for_write: bool = False) -> Path:
        # Relative paths resolve against the primary root; absolute/`~` paths are taken as-is and
        # validated against the root set, so the agent can reach a non-primary root by abs path.
        p = Path(path).expanduser()
        candidate = p.resolve() if p.is_absolute() else (self.root / p).resolve()
        root = self._root_for(candidate)
        if root is None:
            raise PermissionError(f"Path escapes allowed roots: {path}")
        if for_write and not root[1]:
            raise PermissionError(f"Path is in a read-only directory: {path}")
        return candidate

    def _resolve_file(self, path: str, *, for_write: bool = False) -> Path:
        file_path = self._resolve(path, for_write=for_write)
        if not file_path.exists():
            raise ValueError(f"File does not exist: {path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        return file_path

    def _relative(self, path: Path) -> str:
        # Display relative to the primary root; files in other roots show their absolute path
        # (unambiguous and round-trippable as an argument back into the tools).
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def _ignored(self, path: Path) -> bool:
        if not self.ignore:
            return False
        root = self._root_for(path)
        if root is None:
            return True
        try:
            relative = path.relative_to(root[0])
        except ValueError:
            return True
        return any(part in self.ignore for part in relative.parts)

    def _search_file(self, path: Path, query: str) -> list[dict[str, object]]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return []

        matches = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if query in line:
                matches.append(
                    {
                        "path": self._relative(path),
                        "line": line_number,
                        "text": line,
                    }
                )
        return matches

    def _resolve_diff_path(self, path: str, *, for_write: bool = False) -> Path:
        return self._resolve(self._clean_diff_path(path), for_write=for_write)

    def _read_patch_lines(self, path: Path) -> list[str]:
        if not path.exists():
            raise ValueError(f"File does not exist: {self._relative(path)}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {self._relative(path)}")
        return path.read_text(encoding="utf-8").splitlines(keepends=True)

    def _parse_unified_diff(self, diff: str) -> list[_FilePatch]:
        lines = diff.splitlines(keepends=True)
        patches = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.startswith("--- "):
                index += 1
                continue
            old_path = self._parse_diff_file_line(line, "---")
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise ValueError("Unified diff is missing +++ file header.")
            new_path = self._parse_diff_file_line(lines[index], "+++")
            index += 1
            hunks = []
            while index < len(lines):
                current = lines[index]
                if current.startswith("--- "):
                    break
                if not current.startswith("@@ "):
                    index += 1
                    continue
                old_start, old_count, new_start, new_count = self._parse_hunk_header(
                    current
                )
                index += 1
                hunk_lines = []
                while index < len(lines):
                    current = lines[index]
                    if current.startswith("@@ ") or current.startswith("--- "):
                        break
                    if current.startswith("\\ No newline at end of file"):
                        index += 1
                        continue
                    if not current or current[0] not in {" ", "+", "-"}:
                        raise ValueError(f"Invalid unified diff hunk line: {current}")
                    hunk_lines.append(current)
                    index += 1
                hunks.append(
                    _Hunk(
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        lines=hunk_lines,
                    )
                )
            if not hunks:
                raise ValueError(f"Unified diff has no hunks for {new_path}.")
            patches.append(
                _FilePatch(old_path=old_path, new_path=new_path, hunks=hunks)
            )
        if not patches:
            raise ValueError("No unified diff file patches found.")
        return patches

    def _parse_diff_file_line(self, line: str, marker: str) -> str:
        value = line[len(marker) :].strip()
        if "\t" in value:
            value = value.split("\t", 1)[0]
        if value == "/dev/null":
            return value
        return self._clean_diff_path(value)

    def _clean_diff_path(self, path: str) -> str:
        if path.startswith("a/") or path.startswith("b/"):
            return path[2:]
        return path

    def _parse_hunk_header(self, line: str) -> tuple[int, int, int, int]:
        parts = line.split()
        if (
            len(parts) < 3
            or not parts[1].startswith("-")
            or not parts[2].startswith("+")
        ):
            raise ValueError(f"Invalid unified diff hunk header: {line}")
        old_start, old_count = self._parse_hunk_range(parts[1][1:])
        new_start, new_count = self._parse_hunk_range(parts[2][1:])
        return old_start, old_count, new_start, new_count

    def _parse_hunk_range(self, value: str) -> tuple[int, int]:
        if "," not in value:
            return int(value), 1
        start, count = value.split(",", 1)
        return int(start), int(count)

    def _apply_hunks(
        self,
        old_lines: list[str],
        hunks: list[_Hunk],
        path: str,
    ) -> list[str]:
        new_lines = []
        cursor = 0
        for hunk in hunks:
            start = max(hunk.old_start - 1, 0)
            if start < cursor:
                raise ValueError(f"Overlapping unified diff hunks for {path}.")
            new_lines.extend(old_lines[cursor:start])
            cursor = start
            old_seen = 0
            new_seen = 0
            for line in hunk.lines:
                prefix = line[0]
                content = line[1:]
                if prefix == " ":
                    if cursor >= len(old_lines) or old_lines[cursor] != content:
                        raise ValueError(f"Unified diff context does not match {path}.")
                    new_lines.append(old_lines[cursor])
                    cursor += 1
                    old_seen += 1
                    new_seen += 1
                elif prefix == "-":
                    if cursor >= len(old_lines) or old_lines[cursor] != content:
                        raise ValueError(f"Unified diff removal does not match {path}.")
                    cursor += 1
                    old_seen += 1
                elif prefix == "+":
                    new_lines.append(content)
                    new_seen += 1
            if old_seen != hunk.old_count or new_seen != hunk.new_count:
                raise ValueError(f"Unified diff hunk line counts do not match {path}.")
        new_lines.extend(old_lines[cursor:])
        return new_lines

    def _apply_patch_chunks(
        self,
        old_lines: list[str],
        chunks: list[list[str]],
        path: str,
    ) -> tuple[list[str], int]:
        new_lines = list(old_lines)
        cursor = 0
        for chunk in chunks:
            old_block = [line[1:] for line in chunk if line[0] in {" ", "-"}]
            new_block = [line[1:] for line in chunk if line[0] in {" ", "+"}]
            if not old_block:
                raise ValueError(
                    f"Patch hunk must include context or removal lines for {path}."
                )
            matches = self._find_line_block_matches(new_lines, old_block, cursor)
            if not matches:
                raise ValueError(f"Patch context does not match {path}.")
            if len(matches) > 1:
                raise ValueError(f"Patch context is ambiguous for {path}.")
            start = matches[0]
            end = start + len(old_block)
            new_lines[start:end] = new_block
            cursor = start + len(new_block)
        return new_lines, len(chunks)

    def _find_line_block_matches(
        self,
        lines: list[str],
        block: list[str],
        start: int,
    ) -> list[int]:
        if not block:
            return [start]
        last = len(lines) - len(block)
        matches = []
        for index in range(start, last + 1):
            if lines[index : index + len(block)] == block:
                matches.append(index)
        return matches
