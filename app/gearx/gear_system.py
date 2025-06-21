from app.gearx.gear import Gear, ExternalGear, InternalGear


class GearSystem:
    def __init__(self, center):
        self.center = center
        self.external_gear_list = []
        self.internal_gear_list = []

    def add_internal_gear(self, gear: ExternalGear):
        self.internal_gear_list.append(gear)
        pass

    def add_external_gear(self, gear: InternalGear):
        self.external_gear_list.append(gear)
        pass
    def optimize_parameters(self):
        pass

    def to_inp(self):
        pass

