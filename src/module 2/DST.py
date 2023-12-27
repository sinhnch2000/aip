class DialogueStateTracker:
    def __init__(self, ontology):
        self.reset_slots(ontology)

    def reset_slots(self, ontology):
        self.slots = {k: None for k, v in ontology.items()}

    def update_slots(self, slots_values):
        for slot, value in slots_values.items():
            if slot in ["price_per_night", "star_rating"] and value != None:
                value = int(value)
            if slot != "number_of_rooms_available":
                self.slots[slot] = value