from __future__ import annotations

from io import BytesIO
import tkinter as tk
from tkinter import ttk
from typing import Callable, Iterable, Sequence
import webbrowser

from PIL import Image, ImageTk

from .deep_zoom_targets import DeepZoomTarget
from .target_thumbnail_data import THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, thumbnail_bytes

ALL_TARGETS = "Alle"
FAVORITES = "Favoriten"


def target_categories(targets: Iterable[DeepZoomTarget]) -> tuple[str, ...]:
    tags = sorted({tag for target in targets for tag in target.tags}, key=str.casefold)
    return (ALL_TARGETS, FAVORITES, *tags)


def filter_deep_zoom_targets(
    targets: Sequence[DeepZoomTarget],
    *,
    query: str = "",
    category: str = ALL_TARGETS,
) -> tuple[DeepZoomTarget, ...]:
    needle = query.strip().casefold()
    category_key = category.casefold()
    matches: list[DeepZoomTarget] = []
    for target in targets:
        if category_key == FAVORITES.casefold() and not target.favorite:
            continue
        if category_key not in {ALL_TARGETS.casefold(), FAVORITES.casefold()} and not any(
            tag.casefold() == category_key for tag in target.tags
        ):
            continue
        haystack = "\n".join((target.name, target.description, *target.tags)).casefold()
        if needle and needle not in haystack:
            continue
        matches.append(target)
    return tuple(matches)


class DeepZoomTargetBrowser(tk.Toplevel):
    """Searchable visual browser for packaged deep-zoom targets."""

    def __init__(
        self,
        parent: tk.Misc,
        targets: Sequence[DeepZoomTarget],
        *,
        on_set_target: Callable[[DeepZoomTarget], None],
        on_load_view: Callable[[DeepZoomTarget], None],
    ) -> None:
        super().__init__(parent)
        self.title("Deep-Zoom-Ziele")
        self.geometry("980x620")
        self.minsize(760, 480)
        self.transient(parent)

        self._targets = tuple(targets)
        self._visible_targets: tuple[DeepZoomTarget, ...] = ()
        self._on_set_target = on_set_target
        self._on_load_view = on_load_view
        self._preview_cache: dict[str, ImageTk.PhotoImage] = {}

        self.search_var = tk.StringVar()
        self.category_var = tk.StringVar(value=ALL_TARGETS)
        self.result_count_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.tags_var = tk.StringVar()
        self.settings_var = tk.StringVar()
        self.coordinates_var = tk.StringVar()
        self.source_var = tk.StringVar()

        self._build_ui()
        self.search_var.trace_add("write", lambda *_: self._refresh_results())
        self.category_var.trace_add("write", lambda *_: self._refresh_results())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._refresh_results()
        self.search_entry.focus_set()

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(12, 10, 12, 8))
        header.pack(fill=tk.X)
        ttk.Label(header, text="Suche").grid(row=0, column=0, sticky="w")
        self.search_entry = ttk.Entry(header, textvariable=self.search_var)
        self.search_entry.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(header, text="Kategorie").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            header,
            textvariable=self.category_var,
            values=target_categories(self._targets),
            state="readonly",
            width=22,
        ).grid(row=1, column=1, sticky="ew")
        ttk.Label(header, textvariable=self.result_count_var, foreground="#555").grid(
            row=1, column=2, sticky="e", padx=(10, 0)
        )
        header.columnconfigure(0, weight=1)

        content = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        list_frame = ttk.Frame(content)
        detail_frame = ttk.Frame(content, padding=(14, 0, 0, 0))
        content.add(list_frame, weight=1)
        content.add(detail_frame, weight=2)

        self.target_list = tk.Listbox(list_frame, exportselection=False, activestyle="dotbox")
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.target_list.yview)
        self.target_list.configure(yscrollcommand=scrollbar.set)
        self.target_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.target_list.bind("<<ListboxSelect>>", self._on_selection_changed)
        self.target_list.bind("<Double-Button-1>", lambda _event: self._load_selected())

        self.preview_label = ttk.Label(detail_frame, anchor="center")
        self.preview_label.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(detail_frame, textvariable=self.title_var, font=("TkDefaultFont", 14, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            detail_frame,
            textvariable=self.description_var,
            wraplength=500,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(8, 8))
        ttk.Label(detail_frame, textvariable=self.tags_var, foreground="#555").pack(anchor="w")
        ttk.Label(detail_frame, textvariable=self.settings_var, wraplength=500, justify=tk.LEFT).pack(
            anchor="w", pady=(8, 0)
        )
        ttk.Label(
            detail_frame,
            textvariable=self.coordinates_var,
            wraplength=500,
            justify=tk.LEFT,
            font=("TkFixedFont", 9),
        ).pack(anchor="w", pady=(8, 0))
        source = ttk.Label(detail_frame, textvariable=self.source_var, foreground="#2457a7", cursor="hand2")
        source.pack(anchor="w", pady=(8, 0))
        source.bind("<Button-1>", lambda _event: self._open_source())

        buttons = ttk.Frame(self, padding=(12, 0, 12, 12))
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Schließen", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Ansicht laden", command=self._load_selected).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        ttk.Button(buttons, text="Mit Übergang hinzufügen", command=self._set_selected_target).pack(
            side=tk.RIGHT, padx=(0, 6)
        )

    def _refresh_results(self) -> None:
        selected = self.selected_target
        selected_id = selected.id if selected else None
        self._visible_targets = filter_deep_zoom_targets(
            self._targets,
            query=self.search_var.get(),
            category=self.category_var.get(),
        )
        self.target_list.delete(0, tk.END)
        for target in self._visible_targets:
            marker = "★ " if target.favorite else ""
            self.target_list.insert(tk.END, marker + target.name)
        self.result_count_var.set(f"{len(self._visible_targets)} Ziele")

        selection_index = 0
        if selected_id:
            for index, target in enumerate(self._visible_targets):
                if target.id == selected_id:
                    selection_index = index
                    break
        if self._visible_targets:
            self.target_list.selection_set(selection_index)
            self.target_list.activate(selection_index)
            self.target_list.see(selection_index)
        self._show_selected_target()

    @property
    def selected_target(self) -> DeepZoomTarget | None:
        selection = self.target_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if index >= len(self._visible_targets):
            return None
        return self._visible_targets[index]

    def _on_selection_changed(self, _event: tk.Event | None = None) -> None:
        self._show_selected_target()

    def _show_selected_target(self) -> None:
        target = self.selected_target
        if target is None:
            self.preview_label.configure(image="", text="Keine passenden Ziele")
            self.title_var.set("")
            self.description_var.set("")
            self.tags_var.set("")
            self.settings_var.set("")
            self.coordinates_var.set("")
            self.source_var.set("")
            return

        preview = self._load_preview(target)
        self.preview_label.configure(image=preview, text="")
        self.title_var.set(target.name)
        self.description_var.set(target.description)
        self.tags_var.set(" · ".join(target.tags))
        self.settings_var.set(
            f"Empfehlung: {target.recommended_iterations} Iterationen, "
            f"{target.reference_bits} Bit, Palette {target.palette}"
        )
        self.coordinates_var.set(
            f"Zentrum: {target.center_x_text}, {target.center_y_text}\n"
            f"Ansichtsbreite: {target.view_width_text}"
        )
        self.source_var.set("Quelle im Browser öffnen")

    def _load_preview(self, target: DeepZoomTarget) -> ImageTk.PhotoImage:
        cached = self._preview_cache.get(target.id)
        if cached is not None:
            return cached
        with BytesIO(thumbnail_bytes(target.id)) as stream:
            image = Image.open(stream).convert("RGB")
            image.load()
        image = image.resize((320, 200), Image.Resampling.LANCZOS)
        preview = ImageTk.PhotoImage(image, master=self)
        self._preview_cache[target.id] = preview
        return preview

    def _set_selected_target(self) -> None:
        target = self.selected_target
        if target is not None:
            self._on_set_target(target)

    def _load_selected(self) -> None:
        target = self.selected_target
        if target is not None:
            self._on_load_view(target)

    def _open_source(self) -> None:
        target = self.selected_target
        if target is not None:
            webbrowser.open(target.source_url)
