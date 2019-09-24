# -*- coding: utf-8 -*-
"""
NMPY - NeuroMatic in Python
Copyright 2019 Jason Rothman
"""
import copy
from NM_Experiment import Experiment
from NM_Utilities import quotes


class Main(object):
    """
    NM Main class
    
    Class information here...
    
    Attributes:
        experiments: list of Experiment objects
        experiment_select: selected Experiment
    """

    def __init__(self):
        self.experiments = []
        self.experiment_select = None
        self.experiment_new(name="", select=True)  # create default Experiment

    def experiment_name_next(self) -> str:
        """
        Create next default name for an Experiment.

        Returns:
            string name if successful, None otherwise
        """
        n = 10 + len(self.experiments)
        for i in range(0, n):
            name = "NMExp" + str(i)
            found = False
            for e in self.experiments:
                if name.casefold() == e.name.casefold():
                    found = True
                    break
            if not found:
                return name
        return None

    def experiment_name_check(self, 
                              name: str, 
                              exists: bool = False, 
                              notexists: bool = False) -> str:
        """
        Check if str name is OK to use for an Experiment.
        This function removes all special characters except '_'.

        Args:
            name: name to check
            exists: check if name already exists as an Experiment
            notexists: check if name does not exist as an Experiment

        Returns:
            string name if successful, None otherwise
        """
        if name is None:
            print("bad Experiment name: None")
            return None
        if len(name) == 0:
            print("bad Experiment name: 0 length")
            return None
        temp = ""
        for e in name:
            if e.isalnum() or e == "_":
                temp += e  # remove special characters
        name = temp
        if len(name) == 0:
            print("bad Experiment name: 0 length")
            return None
        if exists:
            for e in self.experiments:
                if name.casefold() == e.name.casefold():
                    print("Experiment "+Main.quotes(name)+" already exists")
                    return None
        if notexists:
            found = False
            for e in self.experiments:
                if name.casefold() == e.name.casefold():
                    found = True
                    break
            if not found:
                print("Experiment " + quotes(name) + " does not exist")
                return None
        return name  # name is OK

    def experiment_new(self, 
                       name: str, 
                       select: bool = True) -> Experiment:
        """
        Create a new Experiment and add to experiments list.

        Args:
            name: name of new Experiment
            select: select this Experiment

        Returns:
            new Experiment if successful, None otherwise
        """
        if name is None or len(name) == 0:
            name = self.experiment_name_next()
        name = self.experiment_name_check(name=name, exists=True)
        if name is None:
            return None
        e = Experiment(name=name)
        self.experiments.append(e)
        print("created Experiment " + quotes(name))
        if self.experiment_select is None or select:
            self.experiment_select = e
            print("selected Experiment " + quotes(name))
        return e
        
    def experiment_copy(self,
                        name: str, 
                        newname: str,
                        select: bool = False) -> Experiment:
        """
        Copy an existing Experiment and add to experiments list.

        Args:
            name: name of Experiment to copy
            newname: name of new Experiment

        Returns:
            new Experiment if successful, None otherwise
        """
        name = self.experiment_name_check(name=name, notexists=True)
        if name is None or len(name) == 0:
            return False
        toCopy = None
        for e in self.experiments:
            if name.casefold() == e.name.casefold():
                toCopy = e
                break
        if toCopy is None:
            return False
        e = copy.deepcopy(toCopy)
        self.experiments.append(e)
        print("copied Experiment " + quotes(name) + " to " + quotes(newname))
        return True
        
    def experiment_kill(self, name: str) -> bool:
        """
        Kill an Experiment (i.e. remove from experiments list).

        Args:
            name: name of Experiment

        Returns:
            True for success, False otherwise
        """
        name = self.experiment_name_check(name=name, notexists=True)
        if name is None or len(name) == 0:
            return False
        kill = None
        for e in self.experiments:
            if name.casefold() == e.name.casefold():
                kill = e
                break
        if kill is None:
            return False
        selected = kill is self.experiment_select
        self.experiments.remove(kill)
        if selected:
            if len(self.experiments) == 0:
                self.experiment_select = None
            else:
                self.experiment_select = self.experiments[0]
        print("killed Experiment " + quotes(name))
        return True

    def experiment_select(self, name: str) -> bool:
        """
        Select Experiment.

        Args:
            name: name of Experiment

        Returns:
            True for success, False otherwise
        """
        name = self.experiment_name_check(name=name, notexists=True)
        if name is None:
            return False
        for e in self.experiments:
            if name.casefold() == e.name.casefold():
                self.experiment_select = e
                return True
        return False
