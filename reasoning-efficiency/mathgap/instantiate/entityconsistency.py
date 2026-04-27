from typing import Dict, List, Tuple
import random
import numpy as np

from mathgap.instantiate.instantiation import Instantiation
from mathgap.instantiate.instantiators import Instantiator

from mathgap.trees import ProofTree
from mathgap.properties import PropertyKey, PropertyType

Entity = Tuple[str, str]

class PartRateAndUnitAwareEntityInstantiator(Instantiator):
    """ Instantiates entities while being aware of their units and also their parts """
    # NOTE: does not currently support entities with units
    def __init__(self, entities_without_units: List[Entity], entities_with_units: Dict[Entity,str], parts_by_whole: Dict[Entity,List[Entity]], super_sub_entities: Dict[Entity,List[Entity]] = {}, enforce_uniqueness: bool = True, 
                 enforce_uniqueness_on_parts: bool = True, prob_pick_standard_ents_as_parts: float = 0.0) -> None:
        assert len(entities_with_units) == 0, "Units not currently supported"
        assert prob_pick_standard_ents_as_parts == 0.0, "Picking standard entity as part not currently supported"

        self.entities_without_units = entities_without_units
        self.parts_by_whole = parts_by_whole
        self.super_sub_entities = super_sub_entities
        self.enforce_uniqueness = enforce_uniqueness 
        self.enforce_uniqueness_on_parts = enforce_uniqueness_on_parts

    def _instantiate(self, tree: ProofTree, instantiation: Instantiation, skip_existing: bool, seed: int) -> Instantiation:
        available_entities = self.entities_without_units.copy()
        available_parts_by_whole = self.parts_by_whole.copy()
        available_super_sub = self.super_sub_entities.copy()

        if self.enforce_uniqueness:
            available_entities = [e for e in available_entities if e not in instantiation._instantiations.values()]
            available_parts_by_whole = {
                whole:parts 
                for whole,parts in available_parts_by_whole.items() 
                if whole not in instantiation._instantiations.values() 
                    and not any([p in instantiation._instantiations.values() for p in parts])
            }
            available_super_sub = {
                sup:[s for s in subs if s not in instantiation._instantiations.values()]
                for sup,subs in available_super_sub.items()
                if sup not in instantiation._instantiations.values() \
                    and len([s for s in subs if s not in instantiation._instantiations.values()]) > 0
            }

        # anaylze entity-specs
        parts_by_entity = {}
        sub_by_super_entity = {}
        for lf,entity_specs in [(lf,lf.get_entity_specs()) for lf in tree.nodes_by_lf.keys()]:
            for es in entity_specs:
                if es.has_unit:
                    raise NotImplementedError("Units not supported")
                if es.has_part_entities:
                    parts_by_entity[es.entity_id] = es.part_entity_ids
                if es.has_super_entity:
                    sub_by_super_entity[es.super_entity_id] = es.entity_id

        instantiated_entities = set([])
        newly_instantiated_entities = set([])

        # instantiate all the rate/super-sub entities
        for super_entity_id, sub_entity_id in sub_by_super_entity.items():
            prop_super_entity = PropertyKey(PropertyType.ENTITY, super_entity_id)
            prop_sub_entity = PropertyKey(PropertyType.ENTITY, sub_entity_id)

            super_entity_name = random.choice(list(available_super_sub.keys()))
            sub_entity_name = random.choice(available_super_sub[super_entity_name])

            if skip_existing:
                if prop_super_entity in instantiation:
                    super_entity_name = instantiation[prop_super_entity]
                    assert super_entity_name in self.super_sub_entities, f"{prop_super_entity} is assigned to {super_entity_name}, which is not a valid super-entity!"
                    if prop_sub_entity in instantiation:
                        sub_entity_name = instantiation[prop_sub_entity]
                        assert sub_entity_name in self.super_sub_entities[super_entity_name], f"super-{prop_super_entity} is assigned to {super_entity_name} and sub-{prop_sub_entity} to {sub_entity_name}, which is not a valid combination!"
                    else:
                        sub_entity_name = random.choice(available_super_sub[super_entity_name])
                        instantiation[prop_sub_entity] = sub_entity_name
                        newly_instantiated_entities.add(prop_sub_entity)
                elif prop_sub_entity in instantiation:
                    sub_entity_name = instantiation[prop_sub_entity]
                    super_entities_with_sub = set([sup for sup,subs in self.super_sub_entities.items() if sub_entity_name in subs])
                    assert len(super_entities_with_sub) > 0, f"Requires at least one super-entity with sub-entity {sub_entity_name}, as it has been pre-initialized"
                    super_entity_name = super_entities_with_sub.pop()
                    instantiation[prop_super_entity] = super_entity_name
                    newly_instantiated_entities.add(prop_super_entity)
                else:
                    assert prop_super_entity not in instantiation, f"Cannot overwrite {prop_super_entity} without violating super-sub relationship! {instantiation[prop_super_entity]}->{super_entity_name}"
                    assert prop_sub_entity not in instantiation, f"Cannot overwrite {prop_sub_entity} without violating super-sub relationship! {instantiation[prop_sub_entity]}->{sub_entity_name}"

                    instantiation.set_even_if_present(prop_super_entity, super_entity_name)
                    instantiation.set_even_if_present(prop_sub_entity, sub_entity_name)

                    newly_instantiated_entities.add(prop_super_entity)
                    newly_instantiated_entities.add(prop_sub_entity)
            else:
                assert prop_super_entity not in instantiation, f"Cannot overwrite {prop_super_entity} without violating super-sub relationship! {instantiation[prop_super_entity]}->{super_entity_name}"
                assert prop_sub_entity not in instantiation, f"Cannot overwrite {prop_sub_entity} without violating super-sub relationship! {instantiation[prop_sub_entity]}->{sub_entity_name}"

                instantiation.set_even_if_present(prop_super_entity, super_entity_name)
                instantiation.set_even_if_present(prop_sub_entity, sub_entity_name)

                newly_instantiated_entities.add(prop_super_entity)
                newly_instantiated_entities.add(prop_sub_entity)

            instantiated_entities.add(super_entity_id)
            instantiated_entities.add(sub_entity_id)
            
            # remove from available entities
            available_entities = [e for e in available_entities if e != super_entity_name and e != sub_entity_name]
            available_parts_by_whole = {
                w:ps
                for w,ps in available_parts_by_whole.items()
                if super_entity_name != w and super_entity_name not in ps \
                    and sub_entity_name != w and sub_entity_name not in ps
            }
            available_super_sub = {
                sup:subs
                for sup,subs in available_super_sub.items()
                if super_entity_name != sup and super_entity_name not in subs \
                    and sub_entity_name != sup and sub_entity_name not in subs
            }


        # instantiate all the part_whole entities
        for entity_id, part_ids in parts_by_entity.items():
            prop_entity = PropertyKey(PropertyType.ENTITY, entity_id)
            entity_name,part_names = random.choice(list(available_parts_by_whole.items()))
            simple_ent = False

            # instantiate the whole-entity
            if skip_existing:
                if prop_entity in instantiation:
                    # if the whole-entity is already assigned, we need to respect the assignment and make sure all parts do too
                    entity_name = instantiation[prop_entity]
                    if not entity_name in self.parts_by_whole:
                        print(f"Warning: {prop_entity} is pre-instantiated with {entity_name}, which does not have any part-entities, using whole=part instead.")
                        simple_ent = True
                elif any(PropertyKey(PropertyType.ENTITY, part_id) in instantiation for part_id in part_ids):
                    # if parts of the whole have been instantiated already, then we need to use that whole!
                    inst_part_names = set([instantiation[PropertyKey(PropertyType.ENTITY, part_id)] for part_id in part_ids if PropertyKey(PropertyType.ENTITY, part_id) in instantiation])
                    wholes_with_part = [whole for whole,parts in self.parts_by_whole.items() if all((ipart in parts) for ipart in inst_part_names)]
                    if len(wholes_with_part) > 0:
                        entity_name = wholes_with_part.pop()
                    else:
                        assert len(inst_part_names) == 1, f"Can only have one pre-instantiated part of a whole if we need to resort to whole=part, but got {inst_part_names}"
                        entity_name = inst_part_names.pop()
                        simple_ent = True
                    instantiation[prop_entity] = entity_name
                    newly_instantiated_entities.add(prop_entity)
                else:
                    # neither the whole entity nor any part is already assigned
                    instantiation[prop_entity] = entity_name
                    newly_instantiated_entities.add(prop_entity)
            else:
                # we overwrite existing entities if they have not been explicitly assigned previously by super-sub instantiation
                if prop_entity in newly_instantiated_entities and prop_entity in instantiation:
                    # if the whole-entity is already assigned, we need to respect the assignment and make sure all parts do too
                    entity_name = instantiation[prop_entity]
                    if not entity_name in self.parts_by_whole:
                        print(f"Warning: {prop_entity} is pre-instantiated with {entity_name}, which does not have any part-entities, using whole=part instead.")
                        simple_ent = True
                elif any(PropertyKey(PropertyType.ENTITY, part_id) in newly_instantiated_entities and PropertyKey(PropertyType.ENTITY, part_id) in instantiation for part_id in part_ids):
                    # if parts of the whole have been instantiated already, then we need to use that whole!
                    inst_part_names = set([instantiation[PropertyKey(PropertyType.ENTITY, part_id)] for part_id in part_ids if PropertyKey(PropertyType.ENTITY, part_id) in instantiation])
                    wholes_with_part = [whole for whole,parts in self.parts_by_whole.items() if all((ipart in parts) for ipart in inst_part_names)]
                    if len(wholes_with_part) > 0:
                        entity_name = wholes_with_part.pop()
                    else:
                        assert len(inst_part_names) == 1, f"Can only have one pre-instantiated part of a whole if we need to resort to whole=part, but got {inst_part_names}"
                        entity_name = inst_part_names.pop()
                        simple_ent = True
                    instantiation[prop_entity] = entity_name
                    newly_instantiated_entities.add(prop_entity)
                else:
                    # neither the whole entity nor any part is already assigned
                    instantiation[prop_entity] = entity_name
                    newly_instantiated_entities.add(prop_entity)
            instantiated_entities.add(entity_id)

            if not simple_ent:
                part_names = self.parts_by_whole[entity_name].copy()
                
                if len(part_names) == 0:
                    # switch to simple whole=part due to lack of parts
                    part_names = [entity_name]
                    simple_ent = True 
            else:
                part_names = [entity_name]
            
            # instantiate all part-entities
            available_part_names = part_names.copy()
            if self.enforce_uniqueness and not simple_ent:
                available_part_names = [p for p in available_part_names if p not in instantiation._instantiations.values()]

            for part_entity_id in part_ids:
                prop_part = PropertyKey(PropertyType.ENTITY, part_entity_id)
                if skip_existing:
                    if prop_part in instantiation:
                        # if the part-entity is pre-instantiated, make sure it's consistent with the whole-entity
                        assert instantiation[prop_part] in part_names, f"Invalid whole <-> part mapping ({entity_name} <-> {instantiation[prop_part]}). This is likely due to partial instantiation of partwhole entities"
                        part_name = instantiation[prop_part]
                    else:
                        assert len(available_part_names) > 0, f"No remaining available parts for {entity_name}"
                        part_name = random.choice(available_part_names)
                        instantiation[prop_part] = part_name
                        newly_instantiated_entities.add(prop_part)
                else:
                    if prop_part in newly_instantiated_entities and prop_part in instantiation:
                        # if the part-entity is pre-instantiated, make sure it's consistent with the whole-entity
                        assert instantiation[prop_part] in part_names, f"Invalid whole <-> part mapping ({entity_name} <-> {instantiation[prop_part]}). This is likely due to partial instantiation of partwhole entities"
                        part_name = instantiation[prop_part]
                    else:
                        assert len(available_part_names) > 0, f"No remaining available parts for {entity_name}, started with {part_names.copy()}"
                        part_name = random.choice(available_part_names)
                        instantiation[prop_part] = part_name
                        newly_instantiated_entities.add(prop_part)
                instantiated_entities.add(part_entity_id)

                if self.enforce_uniqueness_on_parts and part_name in available_part_names and not simple_ent:
                    available_part_names.remove(part_name) 

            if self.enforce_uniqueness and entity_name in available_parts_by_whole:
                available_parts_by_whole.pop(entity_name)
            if self.enforce_uniqueness and entity_name in available_entities:
                available_entities.remove(entity_name)

        # then instantiate the rest
        for entity_id in tree.property_tracker.get_by_type(PropertyType.ENTITY):
            prop_entity = PropertyKey(PropertyType.ENTITY, entity_id)
            if entity_id in instantiated_entities: continue # skip the entities that we've already instantiated
            if skip_existing and prop_entity in instantiation: continue

            assert len(available_entities) > 0, "Need at least one entity to choose from! Maybe you are enforcing uniqueness and the list of provided entities is too short?"
            entity_name = random.choice(available_entities)
            
            instantiation[prop_entity] = entity_name
            newly_instantiated_entities.add(prop_entity)
            instantiated_entities.add(entity_id)

            if self.enforce_uniqueness:
                if entity_name in available_entities: available_entities.remove(entity_name)

        assert all(PropertyKey(PropertyType.ENTITY, entity_id) in instantiation._instantiations for entity_id in tree.property_tracker.get_by_type(PropertyType.ENTITY)), f"All entities should have been instantiated! But missing ({[entity_id for entity_id in tree.property_tracker.get_by_type(PropertyType.ENTITY) if PropertyKey(PropertyType.ENTITY, entity_id) not in instantiation._instantiations]})"

        return instantiation

class EntityAwareUnitInstantiator(Instantiator):
    """ 
        Instantiates units while being aware of the entities it's used with
        NOTE: this instantiator can only be called after all entities have been instantiated already
    """
    def __init__(self, unit_by_entity: Dict[str, str]) -> None:
        self.entities_with_units = unit_by_entity

    def _instantiate(self, tree: ProofTree, instantiation: Instantiation, skip_existing: bool, seed: int) -> Instantiation:
        # unit -> entity mapping
        entity_by_unit = {}
        for entity_specs in [lf.get_entity_specs() for lf in tree.nodes_by_lf.keys()]:
            for es in entity_specs:
                if es.has_unit:
                    entity_by_unit[es.unit_id] = es.entity_id

        # instantiate
        for unit_id in tree.property_tracker.get_by_type(PropertyType.UNIT):
            prop_unit = PropertyKey(PropertyType.UNIT, unit_id)
            entity_id = entity_by_unit[unit_id]
            prop_entity = PropertyKey(PropertyType.ENTITY, entity_id)
            assert prop_entity in instantiation, "Instantiator expects all entities that are used with units to be initialized already"
            entity_name = instantiation[prop_entity]
            
            if skip_existing and prop_unit in instantiation: 
                assert instantiation[prop_unit] == self.entities_with_units[entity_name], f"Inconsistent entity <-> unit mapping ({entity_name} <-> {instantiation[prop_unit]}) detected! This is likely due to partial instantiation of the units."

            instantiation[prop_unit] = self.entities_with_units[entity_name]

        return instantiation