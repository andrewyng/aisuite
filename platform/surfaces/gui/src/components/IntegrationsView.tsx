import { ConnectorsTab, McpTab } from "./ManageModal";
import { Icon } from "./Icon";

// Combined "Integrations" surface: messaging connectors + MCP servers in one place.
export function IntegrationsView() {
  return (
    <div className="main page-view">
      <div className="page-col">
        <div className="sa-view-head">
          <div className="sa-view-heading">
            <div className="sa-view-title"><Icon name="plug" size={21} /> Integrations</div>
            <div className="sa-view-sub">Connect the apps and tools OpenCoworker can use.</div>
          </div>
        </div>
        <div className="main-scroll">
          <div className="page-panel">
            <div className="sa-sub">Connectors</div>
            <ConnectorsTab />
            <div className="sa-sub" style={{ marginTop: 26 }}>
              MCP servers
            </div>
            <McpTab />
          </div>
        </div>
      </div>
    </div>
  );
}
