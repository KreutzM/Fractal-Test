from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mpmath as mp

from .camera import CameraState
from .flight_path import CameraPath, CenterInterpolation, Easing, FlightKeyframe
from .flight_plan import (
    FlightPlanDocument,
    FlightScene,
    RenderProfile,
    RenderTrack,
)
from .path_editor import CameraPathDraft
from .surface_lighting import SurfaceLightingSettings

if TYPE_CHECKING:
    from .flight_transition import TransitionPlan


SessionListener = Callable[["FlightPlanSession"], None]


class FlightPlanSession:
    """Tk-independent mutable owner of one editable flight-plan document."""

    def __init__(
        self,
        *,
        camera_draft: CameraPathDraft,
        scene: FlightScene = FlightScene(),
        render_track: RenderTrack | None = None,
        surface_lighting: SurfaceLightingSettings = SurfaceLightingSettings(),
        name: str = "Unbenannter Flugplan",
        file_path: Path | None = None,
        dirty: bool = False,
    ) -> None:
        self._camera_draft = camera_draft
        self._scene = scene
        self._render_track = render_track or RenderTrack.default(digits=camera_draft.digits)
        if not isinstance(surface_lighting, SurfaceLightingSettings):
            raise ValueError("surface_lighting must be SurfaceLightingSettings")
        self._surface_lighting = surface_lighting
        if self._render_track.digits != camera_draft.digits:
            raise ValueError("camera draft and render track must use the same precision")
        self._name = name
        self._file_path = file_path
        self._dirty = bool(dirty)
        self._selected_keyframe_index: int | None = None
        self._playhead_time_text = "0"
        self._listeners: list[SessionListener] = []
        self._validated_document_name()

    @classmethod
    def new(
        cls,
        camera: CameraState,
        *,
        digits: int,
        scene: FlightScene = FlightScene(),
        render_profile: RenderProfile = RenderProfile(),
        surface_lighting: SurfaceLightingSettings = SurfaceLightingSettings(),
        name: str = "Unbenannter Flugplan",
    ) -> "FlightPlanSession":
        draft = CameraPathDraft(digits=digits).add_keyframe(
            "0",
            camera,
            Easing.SMOOTHSTEP,
            CenterInterpolation.FOCUS,
        )
        return cls(
            camera_draft=draft,
            scene=scene,
            render_track=RenderTrack.default(render_profile, digits=digits),
            surface_lighting=surface_lighting,
            name=name,
        )

    @property
    def camera_draft(self) -> CameraPathDraft:
        return self._camera_draft

    @property
    def camera_path(self) -> CameraPath | None:
        if not self._camera_draft.valid:
            return None
        return self._camera_draft.build_path()

    @property
    def scene(self) -> FlightScene:
        return self._scene

    @property
    def render_track(self) -> RenderTrack:
        return self._render_track

    @property
    def surface_lighting(self) -> SurfaceLightingSettings:
        return self._surface_lighting

    @property
    def name(self) -> str:
        return self._name

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def selected_keyframe_index(self) -> int | None:
        return self._selected_keyframe_index

    @property
    def playhead_time_text(self) -> str:
        return self._playhead_time_text

    @property
    def validation_error(self) -> str | None:
        error = self._camera_draft.validation_error
        if error is not None:
            return error
        try:
            self.build_document()
        except ValueError as exc:
            return str(exc)
        return None

    @property
    def valid(self) -> bool:
        return self.validation_error is None

    def build_document(self, *, name: str | None = None) -> FlightPlanDocument:
        path = self._camera_draft.build_path()
        return FlightPlanDocument(
            self._name if name is None else name,
            path,
            self._scene,
            self._render_track,
            self._surface_lighting,
        )

    def add_listener(self, listener: SessionListener, *, notify: bool = False) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)
        if notify:
            listener(self)

    def remove_listener(self, listener: SessionListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:
            pass

    def set_camera_draft(self, draft: CameraPathDraft, *, mark_dirty: bool = True) -> None:
        if draft.digits != self._render_track.digits:
            raise ValueError("camera draft and render track must use the same precision")
        if draft == self._camera_draft:
            return
        self._camera_draft = draft
        self._selected_keyframe_index = _bounded_selection(
            self._selected_keyframe_index,
            len(draft.keyframes),
        )
        if mark_dirty:
            self._dirty = True
        self._notify()

    def set_camera_path(self, path: CameraPath, *, mark_dirty: bool = True) -> None:
        render_track = self._render_track
        if render_track.digits != path.digits:
            render_track = RenderTrack(render_track.cues, digits=path.digits)
            self._render_track = render_track
        self.set_camera_draft(CameraPathDraft.from_path(path), mark_dirty=mark_dirty)

    def set_document(
        self,
        document: FlightPlanDocument,
        *,
        file_path: Path | None = None,
        dirty: bool = False,
    ) -> None:
        self._camera_draft = CameraPathDraft.from_path(document.path)
        self._scene = document.scene
        self._render_track = document.render_track
        self._surface_lighting = document.surface_lighting
        self._name = document.name
        self._file_path = file_path
        self._dirty = bool(dirty)
        self._selected_keyframe_index = 0
        self._playhead_time_text = "0"
        self._notify()

    def append_transition(
        self,
        plan: "TransitionPlan",
        *,
        initialize_if_needed: bool = False,
    ) -> None:
        """Atomically append one generated camera/render transition.

        ``initialize_if_needed`` is used by the right-click workflow.  A fresh
        one-keyframe draft is rebased to the exact camera from which the user
        clicked, but only when the proposal is accepted.  Cancelling the dialog
        therefore never dirties or changes the shared flight plan.
        """

        from .flight_transition import end_render_profile, merge_render_cues

        if plan.digits != self._camera_draft.digits:
            raise ValueError("transition and flight plan must use the same precision")

        draft = self._camera_draft
        track = self._render_track
        scene = self._scene
        if initialize_if_needed and not draft.valid:
            if len(draft.keyframes) > 1:
                raise ValueError("repair the invalid flight-plan draft before adding a target")
            with mp.workdps(plan.digits):
                if mp.mpf(plan.start_time_text) != 0:
                    raise ValueError("a new flight plan must start at zero seconds")
            draft = CameraPathDraft(
                (
                    FlightKeyframe(
                        "0",
                        plan.source_camera,
                        Easing.SMOOTHERSTEP,
                        CenterInterpolation.FOCUS,
                    ),
                ),
                digits=plan.digits,
            )
            track = RenderTrack.default(plan.source_profile, digits=plan.digits)
            scene = plan.scene
        else:
            if not draft.keyframes:
                raise ValueError("a transition requires an existing source keyframe")
            if scene != plan.scene:
                raise ValueError("transition scene does not match the flight plan")
            with mp.workdps(plan.digits):
                current_end = draft.keyframes[-1].time_seconds(digits=plan.digits)
                requested_start = mp.mpf(plan.start_time_text)
                if current_end != requested_start:
                    raise ValueError("transition must start at the current path end")
            if draft.keyframes[-1].camera != plan.source_camera:
                raise ValueError("transition source camera is stale; rebuild the proposal")
            current_profile = end_render_profile(
                render_state=track.evaluate(plan.start_time_text)
            )
            if (
                current_profile.max_iterations != plan.source_profile.max_iterations
                or current_profile.reference_bits != plan.source_profile.reference_bits
                or current_profile.palette != plan.source_profile.palette
                or mp.mpf(current_profile.cycles_text)
                != mp.mpf(plan.source_profile.cycles_text)
            ):
                raise ValueError("transition source render profile is stale; rebuild the proposal")

        for frame in plan.keyframes:
            draft = draft.add_keyframe(
                frame.time_seconds_text,
                frame.camera,
                frame.easing,
                frame.center_interpolation,
            )
        track = RenderTrack(
            merge_render_cues(
                track.cues,
                plan.render_cues,
                digits=plan.digits,
            ),
            digits=plan.digits,
        )
        # Validate the complete candidate before mutating the shared session.
        path = draft.build_path()
        FlightPlanDocument(
            self._name,
            path,
            scene,
            track,
            self._surface_lighting,
        )

        self._camera_draft = draft
        self._scene = scene
        self._render_track = track
        self._dirty = True
        with mp.workdps(plan.digits):
            arrival = mp.mpf(plan.arrival_time_text)
            self._selected_keyframe_index = next(
                index
                for index, frame in enumerate(draft.keyframes)
                if frame.time_seconds(digits=plan.digits) == arrival
            )
        self._playhead_time_text = plan.start_time_text
        self._notify()

    def set_name(self, name: str, *, mark_dirty: bool = True) -> None:
        if name == self._name:
            return
        original = self._name
        self._name = name
        try:
            self._validated_document_name()
        except Exception:
            self._name = original
            raise
        if mark_dirty:
            self._dirty = True
        self._notify()

    def set_scene(self, scene: FlightScene, *, mark_dirty: bool = True) -> None:
        if scene == self._scene:
            return
        self._scene = scene
        if mark_dirty:
            self._dirty = True
        self._notify()

    def set_render_track(self, track: RenderTrack, *, mark_dirty: bool = True) -> None:
        if track.digits != self._camera_draft.digits:
            raise ValueError("camera draft and render track must use the same precision")
        if track == self._render_track:
            return
        self._render_track = track
        if mark_dirty:
            self._dirty = True
        self._notify()

    def set_surface_lighting(
        self,
        settings: SurfaceLightingSettings,
        *,
        mark_dirty: bool = True,
    ) -> None:
        if not isinstance(settings, SurfaceLightingSettings):
            raise ValueError("surface_lighting must be SurfaceLightingSettings")
        if settings == self._surface_lighting:
            return
        self._surface_lighting = settings
        if mark_dirty:
            self._dirty = True
        self._notify()

    def sync_primary_settings(
        self,
        scene: FlightScene,
        profile: RenderProfile,
        surface_lighting: SurfaceLightingSettings | None = None,
        *,
        mark_dirty: bool = True,
    ) -> None:
        track = self._render_track.replace_first_profile(profile)
        lighting = (
            self._surface_lighting
            if surface_lighting is None
            else surface_lighting
        )
        if not isinstance(lighting, SurfaceLightingSettings):
            raise ValueError("surface_lighting must be SurfaceLightingSettings")
        if (
            scene == self._scene
            and track == self._render_track
            and lighting == self._surface_lighting
        ):
            return
        self._scene = scene
        self._render_track = track
        self._surface_lighting = lighting
        if mark_dirty:
            self._dirty = True
        self._notify()

    def set_selected_keyframe(self, index: int | None) -> None:
        bounded = _bounded_selection(index, len(self._camera_draft.keyframes))
        if bounded == self._selected_keyframe_index:
            return
        self._selected_keyframe_index = bounded
        self._notify()

    def set_playhead(self, time_seconds_text: str) -> None:
        if time_seconds_text == self._playhead_time_text:
            return
        path = self.camera_path
        if path is not None:
            path.evaluate(time_seconds_text)
        self._playhead_time_text = time_seconds_text
        self._notify()

    def mark_saved(self, file_path: Path, *, name: str | None = None) -> None:
        if name is not None:
            document = self.build_document(name=name)
            self._name = document.name
        else:
            self.build_document()
        self._file_path = file_path
        self._dirty = False
        self._notify()

    def mark_dirty(self) -> None:
        if self._dirty:
            return
        self._dirty = True
        self._notify()

    def _validated_document_name(self) -> None:
        if not isinstance(self._name, str):
            raise ValueError("flight-plan name must be a string")
        normalized = self._name.strip()
        if not normalized:
            raise ValueError("flight-plan name must not be empty")
        if len(normalized) > 200:
            raise ValueError("flight-plan name must not exceed 200 characters")
        if any(ord(character) < 32 for character in normalized):
            raise ValueError("flight-plan name must not contain control characters")
        self._name = normalized
        path = self.camera_path
        if path is not None:
            FlightPlanDocument(
                self._name,
                path,
                self._scene,
                self._render_track,
                self._surface_lighting,
            )

    def _notify(self) -> None:
        for listener in tuple(self._listeners):
            listener(self)



def _bounded_selection(index: int | None, length: int) -> int | None:
    if index is None or length <= 0:
        return None
    return min(max(int(index), 0), length - 1)
