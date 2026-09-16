"""Selectable deterministic schedulers using shared Harness policies and storage."""
from typing import TypedDict
from .runtime import Harness

async def run_selected(engine, store, registry, session_id, tenant_id, max_steps=None):
    if max_steps is not None and max_steps < 1:
        raise ValueError('max_steps must be positive')
    if engine == 'harness':
        return await Harness(store, registry).run(session_id, tenant_id, max_steps=max_steps)
    from .sdk_controller import SDKController, advance
    limit = max_steps if max_steps is not None else 200
    if engine == 'langgraph':
        from langgraph.graph import StateGraph, START, END
        controller = make_graph_controller(StateGraph, START, END, advance)(store, registry)
        return await controller.run(session_id, tenant_id, max_steps=limit)
    if engine != 'sdk':
        raise ValueError('Unknown runtime')
    from agents.exceptions import UserError
    controller = SDKController(store, registry)
    try:
        return await controller.run(session_id, tenant_id, max_steps=limit)
    except (UserError, RuntimeError) as exc:
        cause = exc.__cause__ if isinstance(exc, UserError) else exc
        if type(cause) is not RuntimeError or str(cause) != 'SDK step budget exhausted' or controller.steps != limit:
            raise
        state = store.load(session_id, tenant_id)
        if state.status != 'running':
            raise
        return state


def make_graph_controller(StateGraph, START, END, advance):
    class GraphState(TypedDict):
        steps: int


    class LangGraphController:
        def __init__(self,store,registry):
            self.store,self.registry=store,registry
            self.policies=Harness(store,registry)
            self.steps=0
        async def run(self,session_id,tenant_id,max_steps=200):
            if max_steps<1:
                raise ValueError('max_steps must be positive')
            self.steps=0
            with self.store.session_lock(session_id):
                state=self.store.load(session_id,tenant_id)
                if state.tool_versions!=self.registry.versions():
                    raise ValueError('Tool versions changed; start a new session')
                if state.status!='running':
                    return state
                graph=StateGraph(GraphState)
                def route(value):
                    if state.status!='running' or value['steps']>=max_steps:
                        return END
                    return state.stage
                def node_for(stage):
                    async def node(value):
                        if state.stage!=stage or state.status!='running':
                            raise ValueError('Invalid stage transition')
                        await advance(self.policies,state)
                        self.steps=value['steps']+1
                        return {'steps':self.steps}
                    return node
                for stage in ('extract','retrieve','verify','correct','finish'):
                    graph.add_node(stage,node_for(stage))
                    graph.add_conditional_edges(stage,route)
                graph.add_conditional_edges(START,route)
                await graph.compile().ainvoke({'steps':0},config={'recursion_limit':max_steps+5})
                return state


    return LangGraphController
