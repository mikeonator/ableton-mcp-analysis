# AbletonMCP/init.py
from __future__ import absolute_import, print_function, unicode_literals

from _Framework.ControlSurface import ControlSurface
import socket
import json
import threading
import time
import traceback
import os

# Change queue import for Python 2
try:
    import Queue as queue  # Python 2
except ImportError:
    import queue  # Python 3

try:
    string_types = (basestring,)
except NameError:
    string_types = (str,)

# Constants for socket communication
DEFAULT_PORT = 9877
HOST = "localhost"

def create_instance(c_instance):
    """Create and return the AbletonMCP script instance"""
    return AbletonMCP(c_instance)

class AbletonMCP(ControlSurface):
    """AbletonMCP Remote Script for Ableton Live"""
    
    def __init__(self, c_instance):
        """Initialize the control surface"""
        ControlSurface.__init__(self, c_instance)
        self.log_message("AbletonMCP Remote Script initializing...")
        
        # Socket server for communication
        self.server = None
        self.client_threads = []
        self.server_thread = None
        self.running = False
        
        # Cache the song reference for easier access
        self._song = self.song()
        
        # Start the socket server
        self.start_server()
        
        self.log_message("AbletonMCP initialized")
        
        # Show a message in Ableton
        self.show_message("AbletonMCP: Listening for commands on port " + str(DEFAULT_PORT))
    
    def disconnect(self):
        """Called when Ableton closes or the control surface is removed"""
        self.log_message("AbletonMCP disconnecting...")
        self.running = False
        
        # Stop the server
        if self.server:
            try:
                self.server.close()
            except:
                pass
        
        # Wait for the server thread to exit
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(1.0)
            
        # Clean up any client threads
        for client_thread in self.client_threads[:]:
            if client_thread.is_alive():
                # We don't join them as they might be stuck
                self.log_message("Client thread still alive during disconnect")
        
        ControlSurface.disconnect(self)
        self.log_message("AbletonMCP disconnected")
    
    def start_server(self):
        """Start the socket server in a separate thread"""
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((HOST, DEFAULT_PORT))
            self.server.listen(5)  # Allow up to 5 pending connections
            
            self.running = True
            self.server_thread = threading.Thread(target=self._server_thread)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.log_message("Server started on port " + str(DEFAULT_PORT))
        except Exception as e:
            self.log_message("Error starting server: " + str(e))
            self.show_message("AbletonMCP: Error starting server - " + str(e))
    
    def _server_thread(self):
        """Server thread implementation - handles client connections"""
        try:
            self.log_message("Server thread started")
            # Set a timeout to allow regular checking of running flag
            self.server.settimeout(1.0)
            
            while self.running:
                try:
                    # Accept connections with timeout
                    client, address = self.server.accept()
                    self.log_message("Connection accepted from " + str(address))
                    self.show_message("AbletonMCP: Client connected")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                    # Keep track of client threads
                    self.client_threads.append(client_thread)
                    
                    # Clean up finished client threads
                    self.client_threads = [t for t in self.client_threads if t.is_alive()]
                    
                except socket.timeout:
                    # No connection yet, just continue
                    continue
                except Exception as e:
                    if self.running:  # Only log if still running
                        self.log_message("Server accept error: " + str(e))
                    time.sleep(0.5)
            
            self.log_message("Server thread stopped")
        except Exception as e:
            self.log_message("Server thread error: " + str(e))
    
    def _handle_client(self, client):
        """Handle communication with a connected client"""
        self.log_message("Client handler started")
        client.settimeout(None)  # No timeout for client socket
        buffer = ''  # Changed from b'' to '' for Python 2
        
        try:
            while self.running:
                try:
                    # Receive data
                    data = client.recv(8192)
                    
                    if not data:
                        # Client disconnected
                        self.log_message("Client disconnected")
                        break
                    
                    # Accumulate data in buffer with explicit encoding/decoding
                    try:
                        # Python 3: data is bytes, decode to string
                        buffer += data.decode('utf-8')
                    except AttributeError:
                        # Python 2: data is already string
                        buffer += data
                    
                    try:
                        # Try to parse command from buffer
                        command = json.loads(buffer)  # Removed decode('utf-8')
                        buffer = ''  # Clear buffer after successful parse
                        
                        self.log_message("Received command: " + str(command.get("type", "unknown")))
                        
                        # Process the command and get response
                        response = self._process_command(command)
                        
                        # Send the response with explicit encoding
                        try:
                            # Python 3: encode string to bytes
                            client.sendall(json.dumps(response).encode('utf-8'))
                        except AttributeError:
                            # Python 2: string is already bytes
                            client.sendall(json.dumps(response))
                    except ValueError:
                        # Incomplete data, wait for more
                        continue
                        
                except Exception as e:
                    self.log_message("Error handling client data: " + str(e))
                    self.log_message(traceback.format_exc())
                    
                    # Send error response if possible
                    error_response = {
                        "status": "error",
                        "message": str(e)
                    }
                    try:
                        # Python 3: encode string to bytes
                        client.sendall(json.dumps(error_response).encode('utf-8'))
                    except AttributeError:
                        # Python 2: string is already bytes
                        client.sendall(json.dumps(error_response))
                    except:
                        # If we can't send the error, the connection is probably dead
                        break
                    
                    # For serious errors, break the loop
                    if not isinstance(e, ValueError):
                        break
        except Exception as e:
            self.log_message("Error in client handler: " + str(e))
        finally:
            try:
                client.close()
            except:
                pass
            self.log_message("Client handler stopped")
    
    def _process_command(self, command):
        """Process a command from the client and return a response"""
        command_type = command.get("type", "")
        params = command.get("params", {})
        
        # Initialize response
        response = {
            "status": "success",
            "result": {}
        }
        
        try:
            # Route the command to the appropriate handler
            if command_type == "get_session_info":
                response["result"] = self._get_session_info()
            elif command_type == "get_track_info":
                track_index = params.get("track_index", 0)
                response["result"] = self._get_track_info(track_index)
            elif command_type == "list_arrangement_clips":
                track_index = params.get("track_index", 0)
                response["result"] = self._list_arrangement_clips(track_index)
            elif command_type == "get_arrangement_clip_source_path":
                track_index = params.get("track_index", 0)
                clip_index = params.get("clip_index", 0)
                response["result"] = self._get_arrangement_clip_source_path(track_index, clip_index)
            elif command_type == "get_detail_clip_source_path":
                response["result"] = self._get_detail_clip_source_path()
            elif command_type == "get_track_devices":
                track_index = params.get("track_index", 0)
                response["result"] = self._get_track_devices(track_index)
            elif command_type == "get_device_parameters":
                track_index = params.get("track_index", 0)
                device_index = params.get("device_index", 0)
                offset = params.get("offset", 0)
                limit = params.get("limit", 64)
                response["result"] = self._get_device_parameters(track_index, device_index, offset, limit)
            elif command_type == "get_transport_state":
                response["result"] = self._get_transport_state()
            elif command_type == "get_tracks_mixer_state":
                response["result"] = self._get_tracks_mixer_state()
            elif command_type == "get_mix_topology":
                include_device_chains = params.get("include_device_chains", True)
                include_device_parameters = params.get("include_device_parameters", False)
                response["result"] = self._get_mix_topology(
                    include_device_chains=include_device_chains,
                    include_device_parameters=include_device_parameters
                )
            elif command_type == "get_automation_envelope_points":
                response["result"] = self._get_automation_envelope_points(
                    track_index=params.get("track_index", 0),
                    scope=params.get("scope", "track_mixer"),
                    mixer_target=params.get("mixer_target", "volume"),
                    send_index=params.get("send_index", None),
                    device_index=params.get("device_index", None),
                    parameter_index=params.get("parameter_index", None),
                    start_time_beats=params.get("start_time_beats", None),
                    end_time_beats=params.get("end_time_beats", None),
                    sample_points=params.get("sample_points", 0)
                )
            # Commands that modify Live's state should be scheduled on the main thread
            elif command_type in ["create_midi_track", "set_track_name", 
                                 "create_clip", "add_notes_to_clip", "set_clip_name", 
                                 "set_tempo", "fire_clip", "stop_clip",
                                 "start_playback", "stop_playback", "load_browser_item",
                                 "load_instrument_or_effect",
                                 "set_transport_state", "set_tracks_mixer_state",
                                 "set_device_parameter"]:
                # Use a thread-safe approach with a response queue
                response_queue = queue.Queue()
                
                # Define a function to execute on the main thread
                def main_thread_task():
                    try:
                        result = None
                        if command_type == "create_midi_track":
                            index = params.get("index", -1)
                            result = self._create_midi_track(index)
                        elif command_type == "set_track_name":
                            track_index = params.get("track_index", 0)
                            name = params.get("name", "")
                            result = self._set_track_name(track_index, name)
                        elif command_type == "create_clip":
                            track_index = params.get("track_index", 0)
                            clip_index = params.get("clip_index", 0)
                            length = params.get("length", 4.0)
                            result = self._create_clip(track_index, clip_index, length)
                        elif command_type == "add_notes_to_clip":
                            track_index = params.get("track_index", 0)
                            clip_index = params.get("clip_index", 0)
                            notes = params.get("notes", [])
                            result = self._add_notes_to_clip(track_index, clip_index, notes)
                        elif command_type == "set_clip_name":
                            track_index = params.get("track_index", 0)
                            clip_index = params.get("clip_index", 0)
                            name = params.get("name", "")
                            result = self._set_clip_name(track_index, clip_index, name)
                        elif command_type == "set_tempo":
                            tempo = params.get("tempo", 120.0)
                            result = self._set_tempo(tempo)
                        elif command_type == "fire_clip":
                            track_index = params.get("track_index", 0)
                            clip_index = params.get("clip_index", 0)
                            result = self._fire_clip(track_index, clip_index)
                        elif command_type == "stop_clip":
                            track_index = params.get("track_index", 0)
                            clip_index = params.get("clip_index", 0)
                            result = self._stop_clip(track_index, clip_index)
                        elif command_type == "start_playback":
                            result = self._start_playback()
                        elif command_type == "stop_playback":
                            result = self._stop_playback()
                        elif command_type == "load_instrument_or_effect":
                            track_index = params.get("track_index", 0)
                            uri = params.get("uri", "")
                            result = self._load_instrument_or_effect(track_index, uri)
                        elif command_type == "load_browser_item":
                            track_index = params.get("track_index", 0)
                            item_uri = params.get("item_uri", "")
                            result = self._load_browser_item(track_index, item_uri)
                        elif command_type == "set_transport_state":
                            song_time_sec = params.get("song_time_sec", None)
                            is_playing = params.get("is_playing", None)
                            result = self._set_transport_state(song_time_sec, is_playing)
                        elif command_type == "set_tracks_mixer_state":
                            states = params.get("states", [])
                            result = self._set_tracks_mixer_state(states)
                        elif command_type == "set_device_parameter":
                            track_index = params.get("track_index", 0)
                            device_index = params.get("device_index", 0)
                            parameter_index = params.get("parameter_index", 0)
                            value = params.get("value", 0.0)
                            result = self._set_device_parameter(
                                track_index, device_index, parameter_index, value
                            )
                        
                        # Put the result in the queue
                        response_queue.put({"status": "success", "result": result})
                    except Exception as e:
                        self.log_message("Error in main thread task: " + str(e))
                        self.log_message(traceback.format_exc())
                        response_queue.put({"status": "error", "message": str(e)})
                
                # Schedule the task to run on the main thread
                try:
                    self.schedule_message(0, main_thread_task)
                except AssertionError:
                    # If we're already on the main thread, execute directly
                    main_thread_task()
                
                # Wait for the response with a timeout
                try:
                    task_response = response_queue.get(timeout=10.0)
                    if task_response.get("status") == "error":
                        response["status"] = "error"
                        response["message"] = task_response.get("message", "Unknown error")
                    else:
                        response["result"] = task_response.get("result", {})
                except queue.Empty:
                    response["status"] = "error"
                    response["message"] = "Timeout waiting for operation to complete"
            elif command_type == "get_browser_item":
                uri = params.get("uri", None)
                path = params.get("path", None)
                response["result"] = self._get_browser_item(uri, path)
            elif command_type == "get_browser_categories":
                category_type = params.get("category_type", "all")
                response["result"] = self._get_browser_categories(category_type)
            elif command_type == "get_browser_items":
                path = params.get("path", "")
                item_type = params.get("item_type", "all")
                response["result"] = self._get_browser_items(path, item_type)
            # Add the new browser commands
            elif command_type == "get_browser_tree":
                category_type = params.get("category_type", "all")
                response["result"] = self.get_browser_tree(category_type)
            elif command_type == "get_browser_items_at_path":
                path = params.get("path", "")
                response["result"] = self.get_browser_items_at_path(path)
            else:
                response["status"] = "error"
                response["message"] = "Unknown command: " + command_type
        except Exception as e:
            self.log_message("Error processing command: " + str(e))
            self.log_message(traceback.format_exc())
            response["status"] = "error"
            response["message"] = str(e)
        
        return response
    
    # Command implementations
    
    def _get_session_info(self):
        """Get information about the current session"""
        try:
            result = {
                "tempo": self._song.tempo,
                "signature_numerator": self._song.signature_numerator,
                "signature_denominator": self._song.signature_denominator,
                "track_count": len(self._song.tracks),
                "return_track_count": len(self._song.return_tracks),
                "is_playing": bool(self._song.is_playing),
                "current_song_time": float(self._song.current_song_time),
                "master_track": {
                    "name": "Master",
                    "volume": self._song.master_track.mixer_device.volume.value,
                    "panning": self._song.master_track.mixer_device.panning.value
                }
            }
            return result
        except Exception as e:
            self.log_message("Error getting session info: " + str(e))
            raise
    
    def _get_track_info(self, track_index):
        """Get information about a track"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            track = self._song.tracks[track_index]
            
            # Get clip slots
            clip_slots = []
            for slot_index, slot in enumerate(track.clip_slots):
                clip_info = None
                if slot.has_clip:
                    clip = slot.clip
                    clip_info = {
                        "name": clip.name,
                        "length": clip.length,
                        "is_playing": clip.is_playing,
                        "is_recording": clip.is_recording
                    }
                
                clip_slots.append({
                    "index": slot_index,
                    "has_clip": slot.has_clip,
                    "clip": clip_info
                })
            
            # Get devices
            devices = []
            for device_index, device in enumerate(track.devices):
                devices.append({
                    "index": device_index,
                    "name": device.name,
                    "class_name": device.class_name,
                    "type": self._get_device_type(device)
                })

            # Some track-like objects can raise on state access (for example arm on non-armable tracks).
            # Keep fields stable but nullable when Live does not expose the property safely.
            is_group_track, track_kind = self._infer_track_kind(track)
            has_audio_input = self._safe_track_state(track, "has_audio_input")
            has_midi_input = self._safe_track_state(track, "has_midi_input")
            group_track_index = self._resolve_group_track_index(self._song, track)
            mute = self._safe_track_state(track, "mute")
            solo = self._safe_track_state(track, "solo")
            arm = self._safe_track_state(track, "arm")

            mixer_device = self._safe_attr(track, "mixer_device")
            volume = self._safe_attr(self._safe_attr(mixer_device, "volume"), "value")
            panning = self._safe_attr(self._safe_attr(mixer_device, "panning"), "value")
            
            result = {
                "index": track_index,
                "name": track.name,
                "is_group_track": is_group_track,
                "track_kind": track_kind,
                "group_track_index": group_track_index,
                "is_audio_track": has_audio_input,
                "is_midi_track": has_midi_input,
                "mute": mute,
                "solo": solo,
                "arm": arm,
                "volume": volume,
                "panning": panning,
                "clip_slots": clip_slots,
                "devices": devices
            }
            return result
        except Exception as e:
            self.log_message("Error getting track info: " + str(e))
            raise

    def _get_song(self):
        """Return the current song object."""
        try:
            song = self.song()
            if song is not None:
                self._song = song
                return song
        except Exception:
            pass
        return self._song

    def _to_list(self, value):
        """Convert Live vectors/iterables into a Python list."""
        if value is None:
            return None
        if isinstance(value, string_types):
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)

        try:
            return list(value)
        except Exception:
            pass

        try:
            length = len(value)
            return [value[index] for index in range(length)]
        except Exception:
            return None

    def _probe_chain(self, root, chain):
        """Probe nested attributes/dict keys defensively."""
        current = root
        for key in chain:
            if current is None:
                return None

            if isinstance(current, dict):
                if key not in current:
                    return None
                current = current.get(key)
                continue

            if not hasattr(current, key):
                return None
            try:
                current = getattr(current, key)
            except Exception:
                return None

        return current

    def _clip_metadata(self, clip, clip_index):
        """Return JSON-safe arrangement clip metadata."""
        metadata = {
            "clip_index": clip_index,
            "clip_name": None,
            "is_audio_clip": None,
            "is_midi_clip": None,
            "start_time_beats": None,
            "end_time_beats": None,
            "length_beats": None
        }

        try:
            metadata["clip_name"] = getattr(clip, "name", None)
        except Exception:
            pass

        try:
            if hasattr(clip, "is_audio_clip"):
                metadata["is_audio_clip"] = bool(getattr(clip, "is_audio_clip"))
        except Exception:
            pass

        try:
            if hasattr(clip, "is_midi_clip"):
                metadata["is_midi_clip"] = bool(getattr(clip, "is_midi_clip"))
        except Exception:
            pass

        try:
            start_value = self._json_scalar(self._safe_attr(clip, "start_time"))
            if isinstance(start_value, (int, float)):
                metadata["start_time_beats"] = float(start_value)
        except Exception:
            pass

        try:
            end_value = self._json_scalar(self._safe_attr(clip, "end_time"))
            if isinstance(end_value, (int, float)):
                metadata["end_time_beats"] = float(end_value)
        except Exception:
            pass

        try:
            length_value = self._json_scalar(self._safe_attr(clip, "length"))
            if isinstance(length_value, (int, float)):
                metadata["length_beats"] = float(length_value)
        except Exception:
            pass

        if (
            isinstance(metadata.get("start_time_beats"), (int, float))
            and isinstance(metadata.get("length_beats"), (int, float))
            and not isinstance(metadata.get("end_time_beats"), (int, float))
        ):
            metadata["end_time_beats"] = float(metadata["start_time_beats"]) + float(metadata["length_beats"])

        return metadata

    def _resolve_file_path_from_clip(self, clip):
        """Best-effort clip source path probing."""
        path_chains = [
            ("clip.file_path", ["file_path"]),
            ("clip.sample.file_path", ["sample", "file_path"]),
            ("clip.sample_path", ["sample_path"]),
            ("clip.path", ["path"]),
            ("clip.sample.filepath", ["sample", "filepath"])
        ]

        file_path = None
        debug_tried = []
        debug_found_at = None

        for chain_name, chain_keys in path_chains:
            debug_tried.append(chain_name)
            probed = self._probe_chain(clip, chain_keys)
            if isinstance(probed, string_types):
                probed = probed.strip()
                if probed:
                    file_path = probed
                    debug_found_at = chain_name
                    break

        exists_on_disk = None
        if file_path:
            try:
                if os.path.isabs(file_path):
                    exists_on_disk = os.path.exists(file_path)
            except Exception:
                exists_on_disk = None

        return {
            "file_path": file_path,
            "exists_on_disk": exists_on_disk,
            "debug_tried": debug_tried,
            "debug_found_at": debug_found_at
        }

    def _list_arrangement_clips(self, track_index):
        """List arrangement clips from Track.arrangement_clips."""
        debug = {
            "hasattr_track_arrangement_clips": False,
            "track_dir_clip_entries": [],
            "exception_type": None,
            "exception_message": None
        }

        try:
            song = self._get_song()
            if song is None:
                raise RuntimeError("Could not access song")

            if track_index < 0 or track_index >= len(song.tracks):
                raise IndexError("Track index out of range")

            track = song.tracks[track_index]
            debug["hasattr_track_arrangement_clips"] = hasattr(track, "arrangement_clips")

            try:
                debug["track_dir_clip_entries"] = [entry for entry in dir(track) if "clip" in entry.lower()]
            except Exception:
                debug["track_dir_clip_entries"] = []

            arrangement_raw = getattr(track, "arrangement_clips")
            arrangement_clips = self._to_list(arrangement_raw)
            if arrangement_clips is None:
                raise TypeError("track.arrangement_clips is not list-like")

            clips = []
            for clip_index, clip in enumerate(arrangement_clips):
                clips.append(self._clip_metadata(clip, clip_index))

            return {
                "supported": True,
                "track_index": track_index,
                "track_name": track.name,
                "arrangement_clip_count": len(arrangement_clips),
                "clips": clips
            }
        except Exception as e:
            debug["exception_type"] = type(e).__name__
            debug["exception_message"] = str(e)
            return {
                "supported": False,
                "track_index": track_index,
                "reason": "arrangement_clip_access_failed",
                "debug": debug
            }

    def _get_arrangement_clip_source_path(self, track_index, clip_index):
        """Get source path for a specific arrangement clip."""
        try:
            song = self._get_song()
            if song is None:
                raise RuntimeError("Could not access song")

            if track_index < 0 or track_index >= len(song.tracks):
                raise IndexError("Track index out of range")

            track = song.tracks[track_index]
            arrangement_clips = self._to_list(getattr(track, "arrangement_clips"))
            if arrangement_clips is None:
                raise TypeError("track.arrangement_clips is not list-like")

            if clip_index < 0 or clip_index >= len(arrangement_clips):
                raise IndexError("Clip index out of range")

            clip = arrangement_clips[clip_index]
            path_info = self._resolve_file_path_from_clip(clip)
            clip_info = self._clip_metadata(clip, clip_index)

            result = {
                "track_index": track_index,
                "track_name": track.name,
                "clip_index": clip_index,
                "clip_name": clip_info.get("clip_name"),
                "is_audio_clip": clip_info.get("is_audio_clip"),
                "is_midi_clip": clip_info.get("is_midi_clip"),
                "file_path": path_info.get("file_path"),
                "exists_on_disk": path_info.get("exists_on_disk"),
                "debug_tried": path_info.get("debug_tried", [])
            }

            if path_info.get("debug_found_at") is not None:
                result["debug_found_at"] = path_info.get("debug_found_at")

            return result
        except Exception as e:
            return {
                "error": "get_arrangement_clip_source_path_failed",
                "track_index": track_index,
                "clip_index": clip_index,
                "message": str(e),
                "exception_type": type(e).__name__
            }

    def _get_detail_clip_source_path(self):
        """Get source path info for the clip currently shown in Detail View."""
        try:
            song = self._get_song()
            if song is None:
                raise RuntimeError("Could not access song")

            detail_clip = getattr(song.view, "detail_clip", None)
            if detail_clip is None:
                return {
                    "error": "no_detail_clip_selected",
                    "message": "No detail clip is currently selected in Ableton Live."
                }

            clip_info = self._clip_metadata(detail_clip, 0)
            path_info = self._resolve_file_path_from_clip(detail_clip)

            result = {
                "clip_name": clip_info.get("clip_name"),
                "is_audio_clip": clip_info.get("is_audio_clip"),
                "is_midi_clip": clip_info.get("is_midi_clip"),
                "file_path": path_info.get("file_path"),
                "exists_on_disk": path_info.get("exists_on_disk"),
                "debug_tried": path_info.get("debug_tried", [])
            }

            if path_info.get("debug_found_at") is not None:
                result["debug_found_at"] = path_info.get("debug_found_at")

            return result
        except Exception as e:
            return {
                "error": "get_detail_clip_source_path_failed",
                "message": str(e),
                "exception_type": type(e).__name__
            }

    def _json_scalar(self, value):
        """Convert values into JSON-safe scalar values."""
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            try:
                if isinstance(value, float):
                    if value != value:
                        return None
                    if value == float("inf") or value == float("-inf"):
                        return None
            except Exception:
                return None
            return value

        if isinstance(value, string_types):
            return value

        try:
            coerced = float(value)
            if coerced != coerced:
                return None
            if coerced == float("inf") or coerced == float("-inf"):
                return None
            return coerced
        except Exception:
            return None

    def _safe_attr(self, obj, attr_name):
        """Best-effort attribute access."""
        if obj is None:
            return None
        try:
            return getattr(obj, attr_name)
        except Exception:
            return None

    def _safe_track_state(self, track, attr_name):
        """Best-effort bool/nullable track-state read for fields like mute/solo/arm."""
        value = self._safe_attr(track, attr_name)
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        try:
            return bool(value)
        except Exception:
            return None

    def _infer_track_kind(self, track):
        """Classify the track as group/audio/midi/hybrid/unknown."""
        is_group_track = self._safe_track_state(track, "is_foldable")
        has_audio_input = self._safe_track_state(track, "has_audio_input")
        has_midi_input = self._safe_track_state(track, "has_midi_input")

        if is_group_track:
            return True, "group"
        if has_audio_input and has_midi_input:
            return False, "hybrid"
        if has_audio_input:
            return False, "audio"
        if has_midi_input:
            return False, "midi"
        return False, "unknown"

    def _resolve_group_track_index(self, song, track):
        """Return parent group track index for a grouped track, else None."""
        if song is None or track is None:
            return None

        group_track = self._safe_attr(track, "group_track")
        if group_track is None:
            return None

        tracks = self._to_list(self._safe_attr(song, "tracks"))
        if not isinstance(tracks, list):
            return None

        for index, candidate in enumerate(tracks):
            try:
                if candidate is group_track or candidate == group_track:
                    return int(index)
            except Exception:
                continue
        return None

    def _set_track_state_bool(self, track, attr_name, value):
        """Best-effort track bool-state write for fields like mute/solo."""
        if track is None or not isinstance(attr_name, string_types):
            return False
        if not hasattr(track, attr_name):
            return False
        try:
            setattr(track, attr_name, bool(value))
            return True
        except Exception:
            return False

    def _get_transport_state(self):
        """Return transport playback and song-time state."""
        song = self._get_song()
        if song is None:
            return {
                "ok": False,
                "error": "song_unavailable",
                "message": "Could not access song"
            }

        return {
            "ok": True,
            "is_playing": bool(self._safe_attr(song, "is_playing")),
            "song_time_sec": float(self._safe_attr(song, "current_song_time") or 0.0)
        }

    def _set_transport_state(self, song_time_sec=None, is_playing=None):
        """Set transport song time and playback state."""
        song = self._get_song()
        if song is None:
            return {
                "ok": False,
                "error": "song_unavailable",
                "message": "Could not access song"
            }

        try:
            if song_time_sec is not None:
                song.current_song_time = max(0.0, float(song_time_sec))
        except Exception as e:
            return {
                "ok": False,
                "error": "invalid_song_time",
                "message": str(e)
            }

        if is_playing is not None:
            try:
                if bool(is_playing):
                    song.start_playing()
                else:
                    song.stop_playing()
            except Exception as e:
                return {
                    "ok": False,
                    "error": "set_playback_failed",
                    "message": str(e)
                }

        transport_state = self._get_transport_state()
        if not isinstance(transport_state, dict):
            transport_state = {}
        transport_state["ok"] = True
        return transport_state

    def _get_tracks_mixer_state(self):
        """Return track solo/mute state snapshot for all normal tracks."""
        song = self._get_song()
        if song is None:
            return {
                "ok": False,
                "error": "song_unavailable",
                "message": "Could not access song"
            }

        states = []
        for track_index, track in enumerate(song.tracks):
            track_name = self._safe_attr(track, "name")
            if track_name is not None and not isinstance(track_name, string_types):
                try:
                    track_name = str(track_name)
                except Exception:
                    track_name = None
            is_group_track, track_kind = self._infer_track_kind(track)
            group_track_index = self._resolve_group_track_index(song, track)

            states.append({
                "track_index": int(track_index),
                "track_name": track_name,
                "is_group_track": is_group_track,
                "track_kind": track_kind,
                "group_track_index": group_track_index,
                "solo": self._safe_track_state(track, "solo"),
                "mute": self._safe_track_state(track, "mute")
            })

        return {
            "ok": True,
            "track_count": len(states),
            "states": states
        }

    def _set_tracks_mixer_state(self, states):
        """Apply solo/mute updates to multiple tracks in one main-thread operation."""
        song = self._get_song()
        if song is None:
            return {
                "ok": False,
                "error": "song_unavailable",
                "message": "Could not access song",
                "updated_count": 0,
                "errors": [{"error": "song_unavailable"}]
            }

        if not isinstance(states, list):
            return {
                "ok": False,
                "error": "invalid_states",
                "message": "states must be a list",
                "updated_count": 0,
                "errors": [{"error": "states_not_list"}]
            }

        updated = []
        errors = []
        for entry in states:
            if not isinstance(entry, dict):
                errors.append({"error": "invalid_entry_type"})
                continue

            track_index = entry.get("track_index")
            try:
                track_index = int(track_index)
            except Exception:
                errors.append({
                    "error": "invalid_track_index",
                    "track_index": track_index
                })
                continue

            if track_index < 0 or track_index >= len(song.tracks):
                errors.append({
                    "error": "track_index_out_of_range",
                    "track_index": track_index
                })
                continue

            track = song.tracks[track_index]
            track_update = {"track_index": track_index}

            if "solo" in entry:
                desired_solo = entry.get("solo")
                if isinstance(desired_solo, bool):
                    if self._set_track_state_bool(track, "solo", desired_solo):
                        track_update["solo"] = bool(desired_solo)
                    else:
                        errors.append({
                            "error": "set_solo_failed",
                            "track_index": track_index
                        })
                elif desired_solo is not None:
                    errors.append({
                        "error": "invalid_solo_value",
                        "track_index": track_index
                    })

            if "mute" in entry:
                desired_mute = entry.get("mute")
                if isinstance(desired_mute, bool):
                    if self._set_track_state_bool(track, "mute", desired_mute):
                        track_update["mute"] = bool(desired_mute)
                    else:
                        errors.append({
                            "error": "set_mute_failed",
                            "track_index": track_index
                        })
                elif desired_mute is not None:
                    errors.append({
                        "error": "invalid_mute_value",
                        "track_index": track_index
                    })

            if len(track_update) > 1:
                updated.append(track_update)

        return {
            "ok": True,
            "updated_count": len(updated),
            "updated": updated,
            "errors": errors
        }

    def _resolve_track_by_index(self, track_index):
        """Resolve and validate a track by index."""
        try:
            track_index = int(track_index)
        except Exception:
            return None, None, {
                "ok": False,
                "error": "invalid_track_index",
                "message": "track_index must be an integer",
                "track_index": track_index
            }

        song = self._get_song()
        if song is None:
            return None, None, {
                "ok": False,
                "error": "song_unavailable",
                "message": "Could not access song",
                "track_index": track_index
            }

        if track_index < 0 or track_index >= len(song.tracks):
            return None, None, {
                "ok": False,
                "error": "invalid_track_index",
                "message": "track_index out of range",
                "track_index": track_index,
                "track_count": len(song.tracks)
            }

        return song, song.tracks[track_index], None

    def _extract_device_plugin_metadata(self, device):
        """Best-effort plugin metadata extraction."""
        class_name = self._safe_attr(device, "class_name")
        if class_name is not None and not isinstance(class_name, string_types):
            try:
                class_name = str(class_name)
            except Exception:
                class_name = None

        is_plugin = None
        plugin_format = None
        vendor = None

        bool_keys = ["is_plugin", "is_plug_in", "is_vst", "is_au"]
        for key in bool_keys:
            value = self._safe_attr(device, key)
            if isinstance(value, bool):
                if key == "is_plugin":
                    is_plugin = value
                elif value:
                    is_plugin = True
                    if key == "is_vst" and plugin_format is None:
                        plugin_format = "VST"
                    elif key == "is_au" and plugin_format is None:
                        plugin_format = "AU"

        if is_plugin is None and isinstance(class_name, string_types):
            class_name_lower = class_name.lower()
            if "plugin" in class_name_lower:
                is_plugin = True

        for key in ["plugin_format", "plug_in_type", "plugin_type", "format"]:
            value = self._safe_attr(device, key)
            if isinstance(value, string_types) and value.strip():
                plugin_format = value.strip()
                break

        for key in ["vendor", "manufacturer", "maker"]:
            value = self._safe_attr(device, key)
            if isinstance(value, string_types) and value.strip():
                vendor = value.strip()
                break

        return class_name, is_plugin, plugin_format, vendor

    def _serialize_device_chain_entry(self, device_index, device):
        """Serialize one device in a track chain."""
        name = self._safe_attr(device, "name")
        if name is not None and not isinstance(name, string_types):
            try:
                name = str(name)
            except Exception:
                name = None

        class_name, is_plugin, plugin_format, vendor = self._extract_device_plugin_metadata(device)
        parameters = self._to_list(self._safe_attr(device, "parameters"))
        parameter_count = len(parameters) if isinstance(parameters, list) else None

        return {
            "device_index": device_index,
            "name": name,
            "class_name": class_name,
            "is_plugin": is_plugin,
            "plugin_format": plugin_format,
            "vendor": vendor,
            "parameter_count": parameter_count
        }

    def _serialize_device_parameter(self, parameter_index, parameter):
        """Serialize one device parameter."""
        name = self._safe_attr(parameter, "name")
        if name is not None and not isinstance(name, string_types):
            try:
                name = str(name)
            except Exception:
                name = None

        value = self._json_scalar(self._safe_attr(parameter, "value"))
        min_value = self._json_scalar(self._safe_attr(parameter, "min"))
        max_value = self._json_scalar(self._safe_attr(parameter, "max"))

        is_quantized = self._safe_attr(parameter, "is_quantized")
        if not isinstance(is_quantized, bool):
            is_quantized = None

        is_enabled = self._safe_attr(parameter, "is_enabled")
        if not isinstance(is_enabled, bool):
            is_enabled = None

        automation_state_raw = self._safe_attr(parameter, "automation_state")
        automation_state = self._json_scalar(automation_state_raw)
        if automation_state is None and automation_state_raw is not None:
            if isinstance(automation_state_raw, string_types):
                automation_state = automation_state_raw
            else:
                try:
                    automation_state = int(automation_state_raw)
                except Exception:
                    automation_state = None

        return {
            "parameter_index": parameter_index,
            "name": name,
            "value": value,
            "min": min_value,
            "max": max_value,
            "is_quantized": is_quantized,
            "is_enabled": is_enabled,
            "automation_state": automation_state
        }

    def _safe_text(self, value):
        """Convert values to JSON-safe text when possible."""
        if value is None:
            return None
        if isinstance(value, string_types):
            text = value.strip()
            return text or None
        try:
            text = str(value).strip()
            return text or None
        except Exception:
            return None

    def _serialize_routing_choice(self, value):
        """Serialize routing selector objects (best effort)."""
        if value is None:
            return None

        for attr_name in ["display_name", "name"]:
            text = self._safe_text(self._safe_attr(value, attr_name))
            if text:
                return text

        if isinstance(value, string_types):
            return self._safe_text(value)

        try:
            return self._safe_text(str(value))
        except Exception:
            return None

    def _serialize_mixer_parameter(self, parameter):
        """Serialize a mixer device parameter value/range/enabled state."""
        if parameter is None:
            return {
                "value": None,
                "min": None,
                "max": None,
                "is_enabled": None,
                "automation_state": None
            }

        value = self._json_scalar(self._safe_attr(parameter, "value"))
        min_value = self._json_scalar(self._safe_attr(parameter, "min"))
        max_value = self._json_scalar(self._safe_attr(parameter, "max"))

        is_enabled = self._safe_attr(parameter, "is_enabled")
        if not isinstance(is_enabled, bool):
            is_enabled = None

        automation_state_raw = self._safe_attr(parameter, "automation_state")
        automation_state = self._json_scalar(automation_state_raw)
        if automation_state is None and automation_state_raw is not None:
            if isinstance(automation_state_raw, string_types):
                automation_state = automation_state_raw
            else:
                try:
                    automation_state = int(automation_state_raw)
                except Exception:
                    automation_state = None

        return {
            "value": value,
            "min": min_value,
            "max": max_value,
            "is_enabled": is_enabled,
            "automation_state": automation_state
        }

    def _serialize_device_chain(self, track_like, include_parameters=False):
        """Serialize a generic track-like device chain."""
        devices_raw = self._safe_attr(track_like, "devices")
        devices = self._to_list(devices_raw)
        if not isinstance(devices, list):
            devices = []

        device_entries = []
        for device_index, device in enumerate(devices):
            row = self._serialize_device_chain_entry(device_index, device)
            if include_parameters:
                parameters_raw = self._safe_attr(device, "parameters")
                parameters = self._to_list(parameters_raw)
                if not isinstance(parameters, list):
                    parameters = []
                row["parameters"] = [
                    self._serialize_device_parameter(parameter_index, parameter)
                    for parameter_index, parameter in enumerate(parameters)
                ]
            device_entries.append(row)
        return device_entries

    def _serialize_track_routing(self, track):
        """Best-effort routing metadata for a track-like object."""
        return {
            "input_type": self._serialize_routing_choice(self._safe_attr(track, "input_routing_type")),
            "input_channel": self._serialize_routing_choice(self._safe_attr(track, "input_routing_channel")),
            "output_type": self._serialize_routing_choice(self._safe_attr(track, "output_routing_type")),
            "output_channel": self._serialize_routing_choice(self._safe_attr(track, "output_routing_channel"))
        }

    def _serialize_track_sends(self, track, return_tracks):
        """Serialize send slots for a track using return-track order."""
        mixer_device = self._safe_attr(track, "mixer_device")
        sends_raw = self._safe_attr(mixer_device, "sends")
        sends = self._to_list(sends_raw)
        if not isinstance(sends, list):
            sends = []

        rows = []
        for send_index, send_param in enumerate(sends):
            param_payload = self._serialize_mixer_parameter(send_param)
            send_name = None
            if isinstance(return_tracks, list) and send_index < len(return_tracks):
                send_name = self._safe_text(self._safe_attr(return_tracks[send_index], "name"))
            if not send_name:
                send_name = self._safe_text(self._safe_attr(send_param, "name"))

            row = {
                "send_index": int(send_index),
                "target_return_index": int(send_index) if isinstance(return_tracks, list) and send_index < len(return_tracks) else None,
                "name": send_name,
                "value": param_payload.get("value"),
                "min": param_payload.get("min"),
                "max": param_payload.get("max"),
                "is_enabled": param_payload.get("is_enabled"),
                "automation_state": param_payload.get("automation_state")
            }
            rows.append(row)
        return rows

    def _serialize_tracklike_mixer_state(self, track_like, include_arm=False):
        """Serialize common mixer state from a track-like object."""
        mixer_device = self._safe_attr(track_like, "mixer_device")
        volume_param = self._safe_attr(mixer_device, "volume")
        panning_param = self._safe_attr(mixer_device, "panning")

        payload = {
            "volume": self._json_scalar(self._safe_attr(volume_param, "value")),
            "panning": self._json_scalar(self._safe_attr(panning_param, "value")),
            "mute": self._safe_track_state(track_like, "mute"),
            "solo": self._safe_track_state(track_like, "solo")
        }
        if include_arm:
            payload["arm"] = self._safe_track_state(track_like, "arm")
        return payload

    def _serialize_track_topology_entry(self, song, track, track_index, return_tracks, include_device_chains, include_device_parameters):
        """Serialize one normal track topology row."""
        is_group_track, track_kind = self._infer_track_kind(track)
        name = self._safe_text(self._safe_attr(track, "name"))
        group_track_index = self._resolve_group_track_index(song, track)

        row = {
            "scope": "track",
            "index": int(track_index),
            "name": name,
            "track_kind": track_kind,
            "is_group_track": bool(is_group_track),
            "group_track_index": group_track_index,
            "mixer": self._serialize_tracklike_mixer_state(track, include_arm=True),
            "sends": self._serialize_track_sends(track, return_tracks),
            "routing": self._serialize_track_routing(track)
        }

        if include_device_chains:
            row["devices"] = self._serialize_device_chain(track, include_parameters=include_device_parameters)
        else:
            row["devices"] = []

        return row

    def _serialize_return_topology_entry(self, track, return_index, include_device_chains, include_device_parameters):
        """Serialize one return track topology row."""
        name = self._safe_text(self._safe_attr(track, "name"))
        row = {
            "scope": "return",
            "index": int(return_index),
            "name": name,
            "mixer": self._serialize_tracklike_mixer_state(track, include_arm=False),
            "routing": self._serialize_track_routing(track)
        }
        if include_device_chains:
            row["devices"] = self._serialize_device_chain(track, include_parameters=include_device_parameters)
        else:
            row["devices"] = []
        return row

    def _serialize_master_topology_entry(self, master_track, include_device_chains, include_device_parameters):
        """Serialize master track topology row."""
        name = self._safe_text(self._safe_attr(master_track, "name")) or "Master"
        mixer_device = self._safe_attr(master_track, "mixer_device")
        row = {
            "scope": "master",
            "name": name,
            "mixer": {
                "volume": self._json_scalar(self._safe_attr(self._safe_attr(mixer_device, "volume"), "value")),
                "panning": self._json_scalar(self._safe_attr(self._safe_attr(mixer_device, "panning"), "value"))
            }
        }
        if include_device_chains:
            row["devices"] = self._serialize_device_chain(master_track, include_parameters=include_device_parameters)
        else:
            row["devices"] = []
        return row

    def _edge_target_from_output_type(self, output_type):
        """Map output routing label to normalized edge target."""
        if not isinstance(output_type, string_types):
            return None
        text = output_type.strip()
        if not text:
            return None
        lowered = text.lower()
        if "master" in lowered:
            return "master"
        return "output:" + text

    def _get_mix_topology(self, include_device_chains=True, include_device_parameters=False):
        """Return normalized routing/bus/send topology for tracks/returns/master."""
        song = self._get_song()
        if song is None:
            return {
                "ok": False,
                "error": "song_unavailable",
                "message": "Could not access song",
                "tracks": [],
                "returns": [],
                "master": None,
                "edges": [],
                "warnings": ["song_unavailable"]
            }

        warnings = []
        try:
            include_device_chains = bool(include_device_chains)
            include_device_parameters = bool(include_device_parameters)

            tracks = self._to_list(self._safe_attr(song, "tracks"))
            if not isinstance(tracks, list):
                tracks = []

            return_tracks = self._to_list(self._safe_attr(song, "return_tracks"))
            if not isinstance(return_tracks, list):
                return_tracks = []

            track_rows = []
            return_rows = []
            edges = []

            for track_index, track in enumerate(tracks):
                try:
                    row = self._serialize_track_topology_entry(
                        song=song,
                        track=track,
                        track_index=track_index,
                        return_tracks=return_tracks,
                        include_device_chains=include_device_chains,
                        include_device_parameters=include_device_parameters
                    )
                    track_rows.append(row)
                except Exception as exc:
                    warnings.append("track_serialize_failed:{0}:{1}".format(track_index, str(exc)))
                    continue

            for row in track_rows:
                track_index = row.get("index")
                group_track_index = row.get("group_track_index")
                if isinstance(track_index, int) and isinstance(group_track_index, int):
                    edges.append({
                        "from": "track:{0}".format(track_index),
                        "to": "track:{0}".format(group_track_index),
                        "kind": "group_membership"
                    })

                sends = row.get("sends", [])
                if isinstance(sends, list) and isinstance(track_index, int):
                    for send in sends:
                        if not isinstance(send, dict):
                            continue
                        send_index = send.get("send_index")
                        target_return_index = send.get("target_return_index")
                        amount = send.get("value")
                        if not isinstance(send_index, int) or not isinstance(target_return_index, int):
                            continue
                        if not isinstance(amount, (int, float)):
                            continue
                        if float(amount) <= 0.0:
                            continue
                        edges.append({
                            "from": "track:{0}".format(track_index),
                            "to": "return:{0}".format(target_return_index),
                            "kind": "send",
                            "send_index": int(send_index),
                            "amount": float(amount)
                        })

                routing = row.get("routing", {})
                output_type = routing.get("output_type") if isinstance(routing, dict) else None
                edge_target = self._edge_target_from_output_type(output_type)
                if edge_target and isinstance(track_index, int):
                    edges.append({
                        "from": "track:{0}".format(track_index),
                        "to": edge_target,
                        "kind": "output_routing"
                    })

            for return_index, return_track in enumerate(return_tracks):
                try:
                    return_row = self._serialize_return_topology_entry(
                        track=return_track,
                        return_index=return_index,
                        include_device_chains=include_device_chains,
                        include_device_parameters=include_device_parameters
                    )
                    return_rows.append(return_row)
                    edges.append({
                        "from": "return:{0}".format(return_index),
                        "to": "master",
                        "kind": "output_routing"
                    })
                except Exception as exc:
                    warnings.append("return_serialize_failed:{0}:{1}".format(return_index, str(exc)))

            master_track = self._safe_attr(song, "master_track")
            master_row = None
            try:
                if master_track is not None:
                    master_row = self._serialize_master_topology_entry(
                        master_track,
                        include_device_chains=include_device_chains,
                        include_device_parameters=include_device_parameters
                    )
                else:
                    warnings.append("master_track_unavailable")
            except Exception as exc:
                warnings.append("master_serialize_failed:{0}".format(str(exc)))

            return {
                "ok": True,
                "session": {
                    "tempo": self._json_scalar(self._safe_attr(song, "tempo")),
                    "signature_numerator": self._json_scalar(self._safe_attr(song, "signature_numerator")),
                    "signature_denominator": self._json_scalar(self._safe_attr(song, "signature_denominator")),
                    "track_count": len(track_rows),
                    "return_track_count": len(return_rows)
                },
                "tracks": track_rows,
                "returns": return_rows,
                "master": master_row,
                "edges": edges,
                "warnings": warnings
            }
        except Exception as e:
            return {
                "ok": False,
                "error": "get_mix_topology_failed",
                "message": str(e),
                "tracks": [],
                "returns": [],
                "master": None,
                "edges": [],
                "warnings": warnings + ["get_mix_topology_failed"]
            }

    def _safe_int_param(self, value):
        """Best-effort integer coercion for command params."""
        try:
            return int(value)
        except Exception:
            return None

    def _safe_float_param(self, value):
        """Best-effort finite float coercion for command params."""
        scalar = self._json_scalar(value)
        if isinstance(scalar, bool):
            return None
        if isinstance(scalar, (int, float)):
            try:
                out = float(scalar)
            except Exception:
                return None
            try:
                if out != out:
                    return None
                if out == float("inf") or out == float("-inf"):
                    return None
            except Exception:
                return None
            return out
        return None

    def _serialize_parameter_automation_state(self, parameter):
        """Serialize parameter automation_state consistently across parameter types."""
        automation_state_raw = self._safe_attr(parameter, "automation_state")
        automation_state = self._json_scalar(automation_state_raw)
        if automation_state is None and automation_state_raw is not None:
            if isinstance(automation_state_raw, string_types):
                automation_state = automation_state_raw
            else:
                try:
                    automation_state = int(automation_state_raw)
                except Exception:
                    automation_state = None
        return automation_state

    def _infer_track_arrangement_time_range_beats(self, track):
        """Best-effort track timeline range from arrangement clips."""
        arrangement_raw = self._safe_attr(track, "arrangement_clips")
        arrangement_clips = self._to_list(arrangement_raw)
        if not isinstance(arrangement_clips, list) or len(arrangement_clips) == 0:
            return None, None

        starts = []
        ends = []
        for clip in arrangement_clips:
            if clip is None:
                continue
            clip_info = self._clip_metadata(clip, 0)
            start_time = clip_info.get("start_time_beats")
            end_time = clip_info.get("end_time_beats")
            if isinstance(start_time, (int, float)):
                starts.append(float(start_time))
            if isinstance(end_time, (int, float)):
                ends.append(float(end_time))

        if not starts and not ends:
            return None, None

        start_out = min(starts) if starts else None
        end_out = max(ends) if ends else None
        if start_out is not None and end_out is not None and end_out < start_out:
            end_out = start_out
        return start_out, end_out

    def _serialize_envelope_point_row(self, point_index, point):
        """Serialize one automation envelope point (best effort)."""
        time_value = None
        point_value = None
        shape_value = None

        if isinstance(point, dict):
            for key in ["time_beats", "time", "beat_time", "x"]:
                time_value = self._safe_float_param(point.get(key))
                if isinstance(time_value, float):
                    break
            for key in ["value", "y"]:
                point_value = self._safe_float_param(point.get(key))
                if isinstance(point_value, float):
                    break
            shape_value = self._safe_text(point.get("shape")) or self._safe_text(point.get("curve"))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            time_value = self._safe_float_param(point[0])
            point_value = self._safe_float_param(point[1])
            if len(point) >= 3:
                shape_value = self._safe_text(point[2])
        else:
            for attr_name in ["time_beats", "time", "beat_time", "x"]:
                time_value = self._safe_float_param(self._safe_attr(point, attr_name))
                if isinstance(time_value, float):
                    break
            for attr_name in ["value", "y"]:
                point_value = self._safe_float_param(self._safe_attr(point, attr_name))
                if isinstance(point_value, float):
                    break
            shape_value = (
                self._safe_text(self._safe_attr(point, "shape"))
                or self._safe_text(self._safe_attr(point, "curve"))
            )

        if not isinstance(time_value, float) or not isinstance(point_value, float):
            return None

        row = {
            "point_index": int(point_index),
            "time_beats": float(time_value),
            "value": float(point_value)
        }
        if shape_value:
            row["shape"] = shape_value
        return row

    def _extract_envelope_points_best_effort(self, envelope):
        """Probe common envelope point containers and return serialized rows."""
        warnings = []
        candidate_names = [
            "points",
            "breakpoints",
            "break_points",
            "envelope_points",
            "automation_points",
            "events",
            "steps",
            "nodes",
            "get_points"
        ]

        for candidate in candidate_names:
            raw = self._safe_attr(envelope, candidate)
            if raw is None:
                continue

            source_kind = "attribute"
            payload = raw
            if callable(raw):
                try:
                    payload = raw()
                    source_kind = "method"
                except Exception as exc:
                    warnings.append("envelope_point_probe_call_failed:{0}:{1}".format(candidate, str(exc)))
                    continue

            rows = None
            if isinstance(payload, (list, tuple)):
                rows = list(payload)
            else:
                rows = self._to_list(payload)

            if not isinstance(rows, list):
                warnings.append("envelope_point_probe_not_list:{0}".format(candidate))
                continue

            point_rows = []
            for point_index, point in enumerate(rows):
                point_row = self._serialize_envelope_point_row(point_index, point)
                if point_row is not None:
                    point_rows.append(point_row)

            if point_rows:
                try:
                    point_rows.sort(key=lambda row: (row.get("time_beats", 0.0), row.get("point_index", 0)))
                except Exception:
                    pass
                return {
                    "point_access_supported": True,
                    "points": point_rows,
                    "source": candidate,
                    "source_kind": source_kind,
                    "warnings": warnings
                }

            warnings.append("envelope_point_probe_empty_or_unserializable:{0}".format(candidate))

        return {
            "point_access_supported": False,
            "points": [],
            "source": None,
            "source_kind": None,
            "warnings": warnings
        }

    def _sample_envelope_values(self, envelope, start_time_beats, end_time_beats, sample_points):
        """Sample envelope values via value_at_time when direct point access is unavailable."""
        value_at_time = self._safe_attr(envelope, "value_at_time")
        if not callable(value_at_time):
            return {
                "supported": False,
                "samples": [],
                "reason": "value_at_time_unavailable"
            }

        start_value = self._safe_float_param(start_time_beats)
        end_value = self._safe_float_param(end_time_beats)
        count_value = self._safe_int_param(sample_points)

        if start_value is None or end_value is None:
            return {
                "supported": False,
                "samples": [],
                "reason": "sample_range_missing"
            }

        if end_value < start_value:
            end_value = start_value

        if count_value is None or count_value <= 1:
            return {
                "supported": False,
                "samples": [],
                "reason": "invalid_sample_points"
            }

        if count_value > 2000:
            count_value = 2000

        samples = []
        denominator = float(max(count_value - 1, 1))
        for idx in range(count_value):
            try:
                ratio = float(idx) / denominator if denominator > 0.0 else 0.0
                time_beats = float(start_value) + ((float(end_value) - float(start_value)) * ratio)
                value = self._safe_float_param(value_at_time(time_beats))
                samples.append({
                    "sample_index": int(idx),
                    "time_beats": float(time_beats),
                    "value": value
                })
            except Exception as exc:
                return {
                    "supported": False,
                    "samples": [],
                    "reason": "value_at_time_failed",
                    "message": str(exc)
                }

        return {
            "supported": True,
            "samples": samples,
            "reason": None
        }

    def _resolve_automation_target_parameter(
        self,
        song,
        track_index,
        scope,
        mixer_target,
        send_index,
        device_index,
        parameter_index
    ):
        """Resolve a parameter object for automation envelope lookup."""
        _, track, error_payload = self._resolve_track_by_index(track_index)
        if error_payload is not None:
            return None, None, error_payload

        scope_text = self._safe_text(scope) or "track_mixer"
        scope_key = scope_text.lower()

        target_payload = {
            "scope": scope_key,
            "mixer_target": None,
            "send_index": None,
            "device_index": None,
            "device_name": None,
            "parameter_index": None,
            "parameter_name": None
        }

        if scope_key in ("track_mixer", "mixer"):
            mixer_device = self._safe_attr(track, "mixer_device")
            if mixer_device is None:
                return track, None, {
                    "ok": False,
                    "error": "mixer_device_unavailable",
                    "message": "Track mixer_device unavailable",
                    "track_index": track_index
                }

            mixer_target_text = (self._safe_text(mixer_target) or "volume").lower()
            if mixer_target_text in ("volume", "gain"):
                parameter = self._safe_attr(mixer_device, "volume")
                target_payload["mixer_target"] = "volume"
            elif mixer_target_text in ("panning", "pan"):
                parameter = self._safe_attr(mixer_device, "panning")
                target_payload["mixer_target"] = "panning"
            elif mixer_target_text in ("send", "sends"):
                target_payload["mixer_target"] = "send"
                send_idx = self._safe_int_param(send_index)
                if send_idx is None or send_idx < 0:
                    return track, None, {
                        "ok": False,
                        "error": "invalid_send_index",
                        "message": "send_index must be a non-negative integer",
                        "track_index": track_index,
                        "send_index": send_index
                    }
                sends = self._to_list(self._safe_attr(mixer_device, "sends"))
                if not isinstance(sends, list):
                    sends = []
                if send_idx >= len(sends):
                    return track, None, {
                        "ok": False,
                        "error": "invalid_send_index",
                        "message": "send_index out of range",
                        "track_index": track_index,
                        "send_index": send_idx,
                        "send_count": len(sends)
                    }
                parameter = sends[send_idx]
                target_payload["send_index"] = int(send_idx)
            else:
                return track, None, {
                    "ok": False,
                    "error": "invalid_mixer_target",
                    "message": "mixer_target must be one of: volume, panning, send",
                    "track_index": track_index,
                    "mixer_target": mixer_target
                }

            target_payload["scope"] = "track_mixer"
            target_payload["parameter_name"] = self._safe_text(self._safe_attr(parameter, "name"))
            return track, parameter, target_payload

        if scope_key in ("device_parameter", "device"):
            device_idx = self._safe_int_param(device_index)
            param_idx = self._safe_int_param(parameter_index)
            if device_idx is None or device_idx < 0:
                return track, None, {
                    "ok": False,
                    "error": "invalid_device_index",
                    "message": "device_index must be a non-negative integer",
                    "track_index": track_index,
                    "device_index": device_index
                }
            if param_idx is None or param_idx < 0:
                return track, None, {
                    "ok": False,
                    "error": "invalid_parameter_index",
                    "message": "parameter_index must be a non-negative integer",
                    "track_index": track_index,
                    "device_index": device_idx,
                    "parameter_index": parameter_index
                }

            devices = self._to_list(self._safe_attr(track, "devices"))
            if not isinstance(devices, list):
                devices = []
            if device_idx >= len(devices):
                return track, None, {
                    "ok": False,
                    "error": "invalid_device_index",
                    "message": "device_index out of range",
                    "track_index": track_index,
                    "device_index": device_idx,
                    "device_count": len(devices)
                }

            device = devices[device_idx]
            parameters = self._to_list(self._safe_attr(device, "parameters"))
            if not isinstance(parameters, list):
                parameters = []
            if param_idx >= len(parameters):
                return track, None, {
                    "ok": False,
                    "error": "invalid_parameter_index",
                    "message": "parameter_index out of range",
                    "track_index": track_index,
                    "device_index": device_idx,
                    "parameter_index": param_idx,
                    "parameter_count": len(parameters)
                }

            parameter = parameters[param_idx]
            target_payload["scope"] = "device_parameter"
            target_payload["device_index"] = int(device_idx)
            target_payload["device_name"] = self._safe_text(self._safe_attr(device, "name"))
            target_payload["parameter_index"] = int(param_idx)
            target_payload["parameter_name"] = self._safe_text(self._safe_attr(parameter, "name"))
            return track, parameter, target_payload

        return track, None, {
            "ok": False,
            "error": "invalid_scope",
            "message": "scope must be 'track_mixer' or 'device_parameter'",
            "track_index": track_index,
            "scope": scope
        }

    def _get_automation_envelope_points(
        self,
        track_index,
        scope="track_mixer",
        mixer_target="volume",
        send_index=None,
        device_index=None,
        parameter_index=None,
        start_time_beats=None,
        end_time_beats=None,
        sample_points=0
    ):
        """Best-effort automation envelope point read for track mixer or device parameters."""
        song = self._get_song()
        track_idx = self._safe_int_param(track_index)
        if song is None:
            return {
                "ok": False,
                "error": "song_unavailable",
                "message": "Could not access song"
            }
        if track_idx is None:
            return {
                "ok": False,
                "error": "invalid_track_index",
                "message": "track_index must be an integer",
                "track_index": track_index
            }

        track, parameter, target_or_error = self._resolve_automation_target_parameter(
            song=song,
            track_index=track_idx,
            scope=scope,
            mixer_target=mixer_target,
            send_index=send_index,
            device_index=device_index,
            parameter_index=parameter_index
        )
        if parameter is None:
            if isinstance(target_or_error, dict):
                return target_or_error
            return {
                "ok": False,
                "error": "automation_target_resolution_failed",
                "message": "Could not resolve automation parameter target",
                "track_index": track_idx
            }

        target_payload = target_or_error if isinstance(target_or_error, dict) else {}
        track_name = self._safe_text(self._safe_attr(track, "name"))
        automation_state = self._serialize_parameter_automation_state(parameter)

        result = {
            "ok": True,
            "supported": True,
            "track_index": int(track_idx),
            "track_name": track_name,
            "target": target_payload,
            "automation_state": automation_state,
            "envelope_exists": False,
            "point_access_supported": False,
            "points": [],
            "sampled_series": [],
            "warnings": []
        }

        envelope_getter = self._safe_attr(song, "automation_envelope")
        if not callable(envelope_getter):
            result["supported"] = False
            result["reason"] = "song_automation_envelope_unavailable"
            result["warnings"].append("song_automation_envelope_unavailable")
            return result

        try:
            envelope = envelope_getter(parameter)
        except Exception as exc:
            result["supported"] = False
            result["reason"] = "automation_envelope_lookup_failed"
            result["message"] = str(exc)
            result["warnings"].append("automation_envelope_lookup_failed")
            return result

        if envelope is None:
            result["envelope_exists"] = False
            return result

        result["envelope_exists"] = True
        result["envelope_class_name"] = self._safe_text(self._safe_attr(envelope, "class_name")) or self._safe_text(type(envelope).__name__)

        direct = self._extract_envelope_points_best_effort(envelope)
        if isinstance(direct, dict):
            result["point_access_supported"] = bool(direct.get("point_access_supported"))
            result["points"] = direct.get("points", []) if isinstance(direct.get("points"), list) else []
            if direct.get("source"):
                result["point_source"] = direct.get("source")
            if direct.get("source_kind"):
                result["point_source_kind"] = direct.get("source_kind")
            if isinstance(direct.get("warnings"), list):
                result["warnings"].extend(direct.get("warnings"))

        sample_count = self._safe_int_param(sample_points)
        if (
            (not isinstance(result.get("points"), list) or len(result.get("points", [])) == 0)
            and isinstance(sample_count, int) and sample_count > 1
        ):
            start_value = self._safe_float_param(start_time_beats)
            end_value = self._safe_float_param(end_time_beats)

            if start_value is None or end_value is None:
                inferred_start, inferred_end = self._infer_track_arrangement_time_range_beats(track)
                if start_value is None:
                    start_value = inferred_start
                if end_value is None:
                    end_value = inferred_end

            sample_payload = self._sample_envelope_values(
                envelope=envelope,
                start_time_beats=start_value,
                end_time_beats=end_value,
                sample_points=sample_count
            )
            if isinstance(sample_payload, dict):
                if bool(sample_payload.get("supported")):
                    result["sampled_series"] = sample_payload.get("samples", [])
                    result["sampled_series_kind"] = "value_at_time"
                    result["sampled_range_beats"] = {
                        "start": start_value,
                        "end": end_value
                    }
                else:
                    sample_reason = sample_payload.get("reason")
                    if sample_reason:
                        result["warnings"].append("sampled_series_unavailable:{0}".format(sample_reason))
                    if sample_payload.get("message"):
                        result["sampled_series_error"] = sample_payload.get("message")

        return result

    def _get_track_devices(self, track_index):
        """Return the ordered device chain for a track."""
        try:
            song, track, error_payload = self._resolve_track_by_index(track_index)
            if error_payload is not None:
                return error_payload

            devices_raw = self._safe_attr(track, "devices")
            devices = self._to_list(devices_raw)
            if not isinstance(devices, list):
                devices = []

            device_entries = []
            for device_index, device in enumerate(devices):
                device_entries.append(self._serialize_device_chain_entry(device_index, device))

            track_name = self._safe_attr(track, "name")
            if track_name is not None and not isinstance(track_name, string_types):
                try:
                    track_name = str(track_name)
                except Exception:
                    track_name = None

            return {
                "ok": True,
                "track_index": int(track_index),
                "track_name": track_name,
                "device_count": len(device_entries),
                "devices": device_entries
            }
        except Exception as e:
            return {
                "ok": False,
                "error": "get_track_devices_failed",
                "message": str(e),
                "track_index": track_index
            }

    def _get_device_parameters(self, track_index, device_index, offset, limit):
        """Return paged parameters for a device on a track."""
        try:
            song, track, error_payload = self._resolve_track_by_index(track_index)
            if error_payload is not None:
                return error_payload

            try:
                device_index = int(device_index)
            except Exception:
                return {
                    "ok": False,
                    "error": "invalid_device_index",
                    "message": "device_index must be an integer",
                    "track_index": track_index,
                    "device_index": device_index
                }

            try:
                offset = int(offset)
            except Exception:
                return {
                    "ok": False,
                    "error": "invalid_offset",
                    "message": "offset must be an integer",
                    "track_index": track_index,
                    "device_index": device_index
                }

            try:
                limit = int(limit)
            except Exception:
                return {
                    "ok": False,
                    "error": "invalid_limit",
                    "message": "limit must be an integer",
                    "track_index": track_index,
                    "device_index": device_index
                }

            if offset < 0:
                return {
                    "ok": False,
                    "error": "invalid_offset",
                    "message": "offset must be >= 0",
                    "track_index": track_index,
                    "device_index": device_index,
                    "offset": offset
                }

            if limit <= 0:
                return {
                    "ok": False,
                    "error": "invalid_limit",
                    "message": "limit must be > 0",
                    "track_index": track_index,
                    "device_index": device_index,
                    "limit": limit
                }

            devices_raw = self._safe_attr(track, "devices")
            devices = self._to_list(devices_raw)
            if not isinstance(devices, list):
                devices = []

            if device_index < 0 or device_index >= len(devices):
                return {
                    "ok": False,
                    "error": "invalid_device_index",
                    "message": "device_index out of range",
                    "track_index": track_index,
                    "device_index": device_index,
                    "device_count": len(devices)
                }

            device = devices[device_index]
            device_name = self._safe_attr(device, "name")
            if device_name is not None and not isinstance(device_name, string_types):
                try:
                    device_name = str(device_name)
                except Exception:
                    device_name = None

            parameters_raw = self._safe_attr(device, "parameters")
            parameters = self._to_list(parameters_raw)
            if not isinstance(parameters, list):
                parameters = []

            total_parameters = len(parameters)
            start_index = min(offset, total_parameters)
            end_index = min(total_parameters, start_index + limit)

            parameter_entries = []
            for parameter_index in range(start_index, end_index):
                parameter_entries.append(
                    self._serialize_device_parameter(parameter_index, parameters[parameter_index])
                )

            return {
                "ok": True,
                "track_index": int(track_index),
                "device_index": int(device_index),
                "device_name": device_name,
                "offset": int(offset),
                "limit": int(limit),
                "total_parameters": total_parameters,
                "parameters": parameter_entries
            }
        except Exception as e:
            return {
                "ok": False,
                "error": "get_device_parameters_failed",
                "message": str(e),
                "track_index": track_index,
                "device_index": device_index
            }

    def _set_device_parameter(self, track_index, device_index, parameter_index, value):
        """Set a parameter value for a device on a normal track."""
        try:
            song, track, error_payload = self._resolve_track_by_index(track_index)
            if error_payload is not None:
                return error_payload

            try:
                device_index = int(device_index)
                parameter_index = int(parameter_index)
            except Exception:
                return {
                    "ok": False,
                    "error": "invalid_index",
                    "message": "device_index and parameter_index must be integers",
                    "track_index": track_index,
                    "device_index": device_index,
                    "parameter_index": parameter_index
                }

            devices = self._to_list(self._safe_attr(track, "devices"))
            if not isinstance(devices, list):
                devices = []
            if device_index < 0 or device_index >= len(devices):
                return {
                    "ok": False,
                    "error": "invalid_device_index",
                    "message": "device_index out of range",
                    "track_index": int(track_index),
                    "device_index": int(device_index),
                    "device_count": len(devices)
                }

            device = devices[device_index]
            parameters = self._to_list(self._safe_attr(device, "parameters"))
            if not isinstance(parameters, list):
                parameters = []
            if parameter_index < 0 or parameter_index >= len(parameters):
                return {
                    "ok": False,
                    "error": "invalid_parameter_index",
                    "message": "parameter_index out of range",
                    "track_index": int(track_index),
                    "device_index": int(device_index),
                    "parameter_index": int(parameter_index),
                    "parameter_count": len(parameters)
                }

            parameter = parameters[parameter_index]
            before_value = self._json_scalar(self._safe_attr(parameter, "value"))
            min_value = self._json_scalar(self._safe_attr(parameter, "min"))
            max_value = self._json_scalar(self._safe_attr(parameter, "max"))

            desired_value = value
            try:
                if not isinstance(desired_value, (int, float)):
                    desired_value = float(desired_value)
            except Exception:
                return {
                    "ok": False,
                    "error": "invalid_value",
                    "message": "value must be numeric",
                    "track_index": int(track_index),
                    "device_index": int(device_index),
                    "parameter_index": int(parameter_index),
                    "value": value
                }

            if isinstance(min_value, (int, float)):
                desired_value = max(float(min_value), float(desired_value))
            if isinstance(max_value, (int, float)):
                desired_value = min(float(max_value), float(desired_value))

            try:
                parameter.value = desired_value
            except Exception as exc:
                return {
                    "ok": False,
                    "error": "set_device_parameter_failed",
                    "message": str(exc),
                    "track_index": int(track_index),
                    "device_index": int(device_index),
                    "parameter_index": int(parameter_index)
                }

            after_value = self._json_scalar(self._safe_attr(parameter, "value"))
            return {
                "ok": True,
                "track_index": int(track_index),
                "device_index": int(device_index),
                "parameter_index": int(parameter_index),
                "device_name": self._safe_text(self._safe_attr(device, "name")),
                "parameter_name": self._safe_text(self._safe_attr(parameter, "name")),
                "before": before_value,
                "after": after_value,
                "min": min_value,
                "max": max_value
            }
        except Exception as e:
            return {
                "ok": False,
                "error": "set_device_parameter_failed",
                "message": str(e),
                "track_index": track_index,
                "device_index": device_index,
                "parameter_index": parameter_index
            }
    
    def _create_midi_track(self, index):
        """Create a new MIDI track at the specified index"""
        try:
            # Create the track
            self._song.create_midi_track(index)
            
            # Get the new track
            new_track_index = len(self._song.tracks) - 1 if index == -1 else index
            new_track = self._song.tracks[new_track_index]
            
            result = {
                "index": new_track_index,
                "name": new_track.name
            }
            return result
        except Exception as e:
            self.log_message("Error creating MIDI track: " + str(e))
            raise
    
    
    def _set_track_name(self, track_index, name):
        """Set the name of a track"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            # Set the name
            track = self._song.tracks[track_index]
            track.name = name
            
            result = {
                "name": track.name
            }
            return result
        except Exception as e:
            self.log_message("Error setting track name: " + str(e))
            raise
    
    def _create_clip(self, track_index, clip_index, length):
        """Create a new MIDI clip in the specified track and clip slot"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            track = self._song.tracks[track_index]
            
            if clip_index < 0 or clip_index >= len(track.clip_slots):
                raise IndexError("Clip index out of range")
            
            clip_slot = track.clip_slots[clip_index]
            
            # Check if the clip slot already has a clip
            if clip_slot.has_clip:
                raise Exception("Clip slot already has a clip")
            
            # Create the clip
            clip_slot.create_clip(length)
            
            result = {
                "name": clip_slot.clip.name,
                "length": clip_slot.clip.length
            }
            return result
        except Exception as e:
            self.log_message("Error creating clip: " + str(e))
            raise
    
    def _add_notes_to_clip(self, track_index, clip_index, notes):
        """Add MIDI notes to a clip"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            track = self._song.tracks[track_index]
            
            if clip_index < 0 or clip_index >= len(track.clip_slots):
                raise IndexError("Clip index out of range")
            
            clip_slot = track.clip_slots[clip_index]
            
            if not clip_slot.has_clip:
                raise Exception("No clip in slot")
            
            clip = clip_slot.clip
            
            # Convert note data to Live's format
            live_notes = []
            for note in notes:
                pitch = note.get("pitch", 60)
                start_time = note.get("start_time", 0.0)
                duration = note.get("duration", 0.25)
                velocity = note.get("velocity", 100)
                mute = note.get("mute", False)
                
                live_notes.append((pitch, start_time, duration, velocity, mute))
            
            # Add the notes
            clip.set_notes(tuple(live_notes))
            
            result = {
                "note_count": len(notes)
            }
            return result
        except Exception as e:
            self.log_message("Error adding notes to clip: " + str(e))
            raise
    
    def _set_clip_name(self, track_index, clip_index, name):
        """Set the name of a clip"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            track = self._song.tracks[track_index]
            
            if clip_index < 0 or clip_index >= len(track.clip_slots):
                raise IndexError("Clip index out of range")
            
            clip_slot = track.clip_slots[clip_index]
            
            if not clip_slot.has_clip:
                raise Exception("No clip in slot")
            
            clip = clip_slot.clip
            clip.name = name
            
            result = {
                "name": clip.name
            }
            return result
        except Exception as e:
            self.log_message("Error setting clip name: " + str(e))
            raise
    
    def _set_tempo(self, tempo):
        """Set the tempo of the session"""
        try:
            self._song.tempo = tempo
            
            result = {
                "tempo": self._song.tempo
            }
            return result
        except Exception as e:
            self.log_message("Error setting tempo: " + str(e))
            raise
    
    def _fire_clip(self, track_index, clip_index):
        """Fire a clip"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            track = self._song.tracks[track_index]
            
            if clip_index < 0 or clip_index >= len(track.clip_slots):
                raise IndexError("Clip index out of range")
            
            clip_slot = track.clip_slots[clip_index]
            
            if not clip_slot.has_clip:
                raise Exception("No clip in slot")
            
            clip_slot.fire()
            
            result = {
                "fired": True
            }
            return result
        except Exception as e:
            self.log_message("Error firing clip: " + str(e))
            raise
    
    def _stop_clip(self, track_index, clip_index):
        """Stop a clip"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            track = self._song.tracks[track_index]
            
            if clip_index < 0 or clip_index >= len(track.clip_slots):
                raise IndexError("Clip index out of range")
            
            clip_slot = track.clip_slots[clip_index]
            
            clip_slot.stop()
            
            result = {
                "stopped": True
            }
            return result
        except Exception as e:
            self.log_message("Error stopping clip: " + str(e))
            raise
    
    
    def _start_playback(self):
        """Start playing the session"""
        try:
            self._song.start_playing()
            
            result = {
                "playing": self._song.is_playing
            }
            return result
        except Exception as e:
            self.log_message("Error starting playback: " + str(e))
            raise
    
    def _stop_playback(self):
        """Stop playing the session"""
        try:
            self._song.stop_playing()
            
            result = {
                "playing": self._song.is_playing
            }
            return result
        except Exception as e:
            self.log_message("Error stopping playback: " + str(e))
            raise
    
    def _get_browser_item(self, uri, path):
        """Get a browser item by URI or path"""
        try:
            # Access the application's browser instance instead of creating a new one
            app = self.application()
            if not app:
                raise RuntimeError("Could not access Live application")
                
            result = {
                "uri": uri,
                "path": path,
                "found": False
            }
            
            # Try to find by URI first if provided
            if uri:
                item = self._find_browser_item_by_uri(app.browser, uri)
                if item:
                    result["found"] = True
                    result["item"] = {
                        "name": item.name,
                        "is_folder": item.is_folder,
                        "is_device": item.is_device,
                        "is_loadable": item.is_loadable,
                        "uri": item.uri
                    }
                    return result
            
            # If URI not provided or not found, try by path
            if path:
                # Parse the path and navigate to the specified item
                path_parts = path.split("/")
                
                # Determine the root based on the first part
                current_item = None
                if path_parts[0].lower() == "nstruments":
                    current_item = app.browser.instruments
                elif path_parts[0].lower() == "sounds":
                    current_item = app.browser.sounds
                elif path_parts[0].lower() == "drums":
                    current_item = app.browser.drums
                elif path_parts[0].lower() == "audio_effects":
                    current_item = app.browser.audio_effects
                elif path_parts[0].lower() == "midi_effects":
                    current_item = app.browser.midi_effects
                else:
                    # Default to instruments if not specified
                    current_item = app.browser.instruments
                    # Don't skip the first part in this case
                    path_parts = ["instruments"] + path_parts
                
                # Navigate through the path
                for i in range(1, len(path_parts)):
                    part = path_parts[i]
                    if not part:  # Skip empty parts
                        continue
                    
                    found = False
                    for child in current_item.children:
                        if child.name.lower() == part.lower():
                            current_item = child
                            found = True
                            break
                    
                    if not found:
                        result["error"] = "Path part '{0}' not found".format(part)
                        return result
                
                # Found the item
                result["found"] = True
                result["item"] = {
                    "name": current_item.name,
                    "is_folder": current_item.is_folder,
                    "is_device": current_item.is_device,
                    "is_loadable": current_item.is_loadable,
                    "uri": current_item.uri
                }
            
            return result
        except Exception as e:
            self.log_message("Error getting browser item: " + str(e))
            self.log_message(traceback.format_exc())
            raise   
    
    
    
    def _load_browser_item(self, track_index, item_uri):
        """Load a browser item onto a track by its URI"""
        try:
            if track_index < 0 or track_index >= len(self._song.tracks):
                raise IndexError("Track index out of range")
            
            track = self._song.tracks[track_index]
            
            # Access the application's browser instance instead of creating a new one
            app = self.application()
            
            # Find the browser item by URI
            item = self._find_browser_item_by_uri(app.browser, item_uri)
            
            if not item:
                raise ValueError("Browser item with URI '{0}' not found".format(item_uri))
            
            # Select the track
            self._song.view.selected_track = track
            
            # Load the item
            app.browser.load_item(item)
            
            result = {
                "loaded": True,
                "item_name": item.name,
                "track_name": track.name,
                "uri": item_uri
            }
            return result
        except Exception as e:
            self.log_message("Error loading browser item: {0}".format(str(e)))
            self.log_message(traceback.format_exc())
            raise
    
    def _find_browser_item_by_uri(self, browser_or_item, uri, max_depth=10, current_depth=0):
        """Find a browser item by its URI"""
        try:
            # Check if this is the item we're looking for
            if hasattr(browser_or_item, 'uri') and browser_or_item.uri == uri:
                return browser_or_item
            
            # Stop recursion if we've reached max depth
            if current_depth >= max_depth:
                return None
            
            # Check if this is a browser with root categories
            if hasattr(browser_or_item, 'instruments'):
                # Check all main categories
                categories = [
                    browser_or_item.instruments,
                    browser_or_item.sounds,
                    browser_or_item.drums,
                    browser_or_item.audio_effects,
                    browser_or_item.midi_effects
                ]
                
                for category in categories:
                    item = self._find_browser_item_by_uri(category, uri, max_depth, current_depth + 1)
                    if item:
                        return item
                
                return None
            
            # Check if this item has children
            if hasattr(browser_or_item, 'children') and browser_or_item.children:
                for child in browser_or_item.children:
                    item = self._find_browser_item_by_uri(child, uri, max_depth, current_depth + 1)
                    if item:
                        return item
            
            return None
        except Exception as e:
            self.log_message("Error finding browser item by URI: {0}".format(str(e)))
            return None
    
    # Helper methods
    
    def _get_device_type(self, device):
        """Get the type of a device"""
        try:
            # Simple heuristic - in a real implementation you'd look at the device class
            if device.can_have_drum_pads:
                return "drum_machine"
            elif device.can_have_chains:
                return "rack"
            elif "instrument" in device.class_display_name.lower():
                return "instrument"
            elif "audio_effect" in device.class_name.lower():
                return "audio_effect"
            elif "midi_effect" in device.class_name.lower():
                return "midi_effect"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def get_browser_tree(self, category_type="all"):
        """
        Get a simplified tree of browser categories.
        
        Args:
            category_type: Type of categories to get ('all', 'instruments', 'sounds', etc.)
            
        Returns:
            Dictionary with the browser tree structure
        """
        try:
            # Access the application's browser instance instead of creating a new one
            app = self.application()
            if not app:
                raise RuntimeError("Could not access Live application")
                
            # Check if browser is available
            if not hasattr(app, 'browser') or app.browser is None:
                raise RuntimeError("Browser is not available in the Live application")
            
            # Log available browser attributes to help diagnose issues
            browser_attrs = [attr for attr in dir(app.browser) if not attr.startswith('_')]
            self.log_message("Available browser attributes: {0}".format(browser_attrs))
            
            result = {
                "type": category_type,
                "categories": [],
                "available_categories": browser_attrs
            }
            
            # Helper function to process a browser item and its children
            def process_item(item, depth=0):
                if not item:
                    return None
                
                result = {
                    "name": item.name if hasattr(item, 'name') else "Unknown",
                    "is_folder": hasattr(item, 'children') and bool(item.children),
                    "is_device": hasattr(item, 'is_device') and item.is_device,
                    "is_loadable": hasattr(item, 'is_loadable') and item.is_loadable,
                    "uri": item.uri if hasattr(item, 'uri') else None,
                    "children": []
                }
                
                
                return result
            
            # Process based on category type and available attributes
            if (category_type == "all" or category_type == "instruments") and hasattr(app.browser, 'instruments'):
                try:
                    instruments = process_item(app.browser.instruments)
                    if instruments:
                        instruments["name"] = "Instruments"  # Ensure consistent naming
                        result["categories"].append(instruments)
                except Exception as e:
                    self.log_message("Error processing instruments: {0}".format(str(e)))
            
            if (category_type == "all" or category_type == "sounds") and hasattr(app.browser, 'sounds'):
                try:
                    sounds = process_item(app.browser.sounds)
                    if sounds:
                        sounds["name"] = "Sounds"  # Ensure consistent naming
                        result["categories"].append(sounds)
                except Exception as e:
                    self.log_message("Error processing sounds: {0}".format(str(e)))
            
            if (category_type == "all" or category_type == "drums") and hasattr(app.browser, 'drums'):
                try:
                    drums = process_item(app.browser.drums)
                    if drums:
                        drums["name"] = "Drums"  # Ensure consistent naming
                        result["categories"].append(drums)
                except Exception as e:
                    self.log_message("Error processing drums: {0}".format(str(e)))
            
            if (category_type == "all" or category_type == "audio_effects") and hasattr(app.browser, 'audio_effects'):
                try:
                    audio_effects = process_item(app.browser.audio_effects)
                    if audio_effects:
                        audio_effects["name"] = "Audio Effects"  # Ensure consistent naming
                        result["categories"].append(audio_effects)
                except Exception as e:
                    self.log_message("Error processing audio_effects: {0}".format(str(e)))
            
            if (category_type == "all" or category_type == "midi_effects") and hasattr(app.browser, 'midi_effects'):
                try:
                    midi_effects = process_item(app.browser.midi_effects)
                    if midi_effects:
                        midi_effects["name"] = "MIDI Effects"
                        result["categories"].append(midi_effects)
                except Exception as e:
                    self.log_message("Error processing midi_effects: {0}".format(str(e)))
            
            # Try to process other potentially available categories
            for attr in browser_attrs:
                if attr not in ['instruments', 'sounds', 'drums', 'audio_effects', 'midi_effects'] and \
                   (category_type == "all" or category_type == attr):
                    try:
                        item = getattr(app.browser, attr)
                        if hasattr(item, 'children') or hasattr(item, 'name'):
                            category = process_item(item)
                            if category:
                                category["name"] = attr.capitalize()
                                result["categories"].append(category)
                    except Exception as e:
                        self.log_message("Error processing {0}: {1}".format(attr, str(e)))
            
            self.log_message("Browser tree generated for {0} with {1} root categories".format(
                category_type, len(result['categories'])))
            return result
            
        except Exception as e:
            self.log_message("Error getting browser tree: {0}".format(str(e)))
            self.log_message(traceback.format_exc())
            raise
    
    def get_browser_items_at_path(self, path):
        """
        Get browser items at a specific path.
        
        Args:
            path: Path in the format "category/folder/subfolder"
                 where category is one of: instruments, sounds, drums, audio_effects, midi_effects
                 or any other available browser category
                 
        Returns:
            Dictionary with items at the specified path
        """
        try:
            # Access the application's browser instance instead of creating a new one
            app = self.application()
            if not app:
                raise RuntimeError("Could not access Live application")
                
            # Check if browser is available
            if not hasattr(app, 'browser') or app.browser is None:
                raise RuntimeError("Browser is not available in the Live application")
            
            # Log available browser attributes to help diagnose issues
            browser_attrs = [attr for attr in dir(app.browser) if not attr.startswith('_')]
            self.log_message("Available browser attributes: {0}".format(browser_attrs))
                
            # Parse the path
            path_parts = path.split("/")
            if not path_parts:
                raise ValueError("Invalid path")
            
            # Determine the root category
            root_category = path_parts[0].lower()
            current_item = None
            
            # Check standard categories first
            if root_category == "instruments" and hasattr(app.browser, 'instruments'):
                current_item = app.browser.instruments
            elif root_category == "sounds" and hasattr(app.browser, 'sounds'):
                current_item = app.browser.sounds
            elif root_category == "drums" and hasattr(app.browser, 'drums'):
                current_item = app.browser.drums
            elif root_category == "audio_effects" and hasattr(app.browser, 'audio_effects'):
                current_item = app.browser.audio_effects
            elif root_category == "midi_effects" and hasattr(app.browser, 'midi_effects'):
                current_item = app.browser.midi_effects
            else:
                # Try to find the category in other browser attributes
                found = False
                for attr in browser_attrs:
                    if attr.lower() == root_category:
                        try:
                            current_item = getattr(app.browser, attr)
                            found = True
                            break
                        except Exception as e:
                            self.log_message("Error accessing browser attribute {0}: {1}".format(attr, str(e)))
                
                if not found:
                    # If we still haven't found the category, return available categories
                    return {
                        "path": path,
                        "error": "Unknown or unavailable category: {0}".format(root_category),
                        "available_categories": browser_attrs,
                        "items": []
                    }
            
            # Navigate through the path
            for i in range(1, len(path_parts)):
                part = path_parts[i]
                if not part:  # Skip empty parts
                    continue
                
                if not hasattr(current_item, 'children'):
                    return {
                        "path": path,
                        "error": "Item at '{0}' has no children".format('/'.join(path_parts[:i])),
                        "items": []
                    }
                
                found = False
                for child in current_item.children:
                    if hasattr(child, 'name') and child.name.lower() == part.lower():
                        current_item = child
                        found = True
                        break
                
                if not found:
                    return {
                        "path": path,
                        "error": "Path part '{0}' not found".format(part),
                        "items": []
                    }
            
            # Get items at the current path
            items = []
            if hasattr(current_item, 'children'):
                for child in current_item.children:
                    item_info = {
                        "name": child.name if hasattr(child, 'name') else "Unknown",
                        "is_folder": hasattr(child, 'children') and bool(child.children),
                        "is_device": hasattr(child, 'is_device') and child.is_device,
                        "is_loadable": hasattr(child, 'is_loadable') and child.is_loadable,
                        "uri": child.uri if hasattr(child, 'uri') else None
                    }
                    items.append(item_info)
            
            result = {
                "path": path,
                "name": current_item.name if hasattr(current_item, 'name') else "Unknown",
                "uri": current_item.uri if hasattr(current_item, 'uri') else None,
                "is_folder": hasattr(current_item, 'children') and bool(current_item.children),
                "is_device": hasattr(current_item, 'is_device') and current_item.is_device,
                "is_loadable": hasattr(current_item, 'is_loadable') and current_item.is_loadable,
                "items": items
            }
            
            self.log_message("Retrieved {0} items at path: {1}".format(len(items), path))
            return result
            
        except Exception as e:
            self.log_message("Error getting browser items at path: {0}".format(str(e)))
            self.log_message(traceback.format_exc())
            raise
