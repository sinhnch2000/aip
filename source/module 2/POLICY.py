from abc import ABC, abstractmethod
from DST import DialogueStateTracker
from utils import *
import sqlite3


class Policy(ABC):
    def __init__(self):
        self.database_slots = ["hotel_name", "destination", "street_address", "number_of_rooms_available",
                               "star_rating", "price_per_night", "has_wifi", "phone_number"]
        self.required_slots_search = ["destination"]
        self.required_slots_book = ["hotel_name", "check_in_date", "number_of_days", "destination", "number_of_rooms"]
        self.required_slots_info = ["hotel_name", "destination"]
        self.system_action = {}
        self.count_search = 0
        self.current_result = {}
        self.current_book = {}
        self.change_slots_after_negate = {}
        self.map_ontology = {"destination": "slot0",
                             "number_of_rooms": "slot1",
                             "check_in_date": "slot2",
                             "number_of_days": "slot3",
                             "star_rating": "slot4",
                             "hotel_name": "slot5",
                             "street_address": "slot6",
                             "phone_number": "slot7",
                             "price_per_night": "slot8",
                             "has_wifi": "slot9"}
        self.tracker = DialogueStateTracker(self.map_ontology)

    def check_slot_to_search(self):
        list_missing_required_slots_search = check_missing_slots(self.tracker.slots, self.required_slots_search)
        if len(list_missing_required_slots_search) == 0:
            return "accept_search"
        else:
            return list_missing_required_slots_search

    def check_slot_to_book(self):
        list_missing_required_slots_book = check_missing_slots(self.tracker.slots, self.required_slots_book)
        if len(list_missing_required_slots_book) == 0:
            return "accept_book"
        else:
            return list_missing_required_slots_book

    def search_hotel(self, alts=False):
        dict_to_search = {}
        for slot, value in self.tracker.slots.items():
            if value != None and slot in self.database_slots:
                dict_to_search.setdefault(slot, value)
        query = generate_query(dict_to_search, self.current_result) if alts else generate_query(dict_to_search)
        rows = select_db(query)
        if rows:
            self.current_result.clear()
            for i in range(len(self.database_slots)):
                self.current_result.setdefault(self.database_slots[i], rows[0][i])
            self.count_search = len(rows)
            print("---result:", self.current_result)
        return dict_to_search

    def search_info(self, slots_values_requested):
        if self.current_result == {}:
            dict_to_search = {"hotel_name":self.tracker.slots["hotel_name"]}
            query = generate_query(dict_to_search)
            rows = select_db(query)
            if rows:
                for i in range(len(self.database_slots)):
                    self.current_result.setdefault(self.database_slots[i], rows[0][i])
                print("---result:", self.current_result)

        inform_value = {}
        for i in slots_values_requested:
            slot = i
            inform_value.setdefault(slot, self.current_result[i])
        return inform_value

    def book_hotel(self):
        for slot in self.database_slots:
            if slot not in ["number_of_rooms_available"]:
                self.current_book[slot] = self.current_result[slot]
        for slot in self.required_slots_book:
            if self.tracker.slots[slot] != None:
                self.current_book[slot] = self.tracker.slots[slot]

    def confirm_book(self):
        confirm_slot = {}
        if self.change_slots_after_negate != {}:
            confirm_slot.update(self.change_slots_after_negate)
            self.change_slots_after_negate.clear()
        else:
            for slot in ["destination", "hotel_name", "check_in_date", "number_of_days", "number_of_rooms"]:
                confirm_slot.setdefault(slot, self.current_book[slot])
        return confirm_slot