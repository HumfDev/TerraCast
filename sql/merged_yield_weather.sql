-- Reference: canonical merged table (default name used by TerraCast).
-- Create from notebook final_df:
--   final_df.write.format("delta").mode("overwrite").saveAsTable("workspace.default.merged")

-- GRANT SELECT ON TABLE workspace.default.merged TO `your-app-principal`;
