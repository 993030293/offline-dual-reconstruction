// Deny remote browser requests while preserving the native complete-asset packer.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.argv[2]);
const launch = chromium.launch.bind(chromium), requests = [], denied = [];
chromium.launch = async options => {
  const browser = await launch({...options,args:[...(options.args??[]),'--disable-background-networking']});
  const createContext = browser.newContext.bind(browser), close = browser.close.bind(browser);
  browser.newContext = async options => {
    const context = await createContext(options);
    await context.route('**/*', route => {
      const url = new URL(route.request().url());
      const allowed = ['127.0.0.1','localhost','[::1]'].includes(url.hostname);
      requests.push({host:url.hostname,path:url.pathname,allowed});
      if(allowed) return route.continue();
      denied.push({host:url.hostname,path:url.pathname});
      return route.abort('blockedbyclient');
    });
    return context;
  };
  browser.close = async () => {
    fs.writeFileSync(path.join(process.argv[4],'OFFLINE_BROWSER_RESULT.json'),JSON.stringify({
      status:denied.length?'external_requests_denied':'only_loopback_requests_observed',requests,denied},null,2)+'\n');
    return close();
  };
  return browser;
};
require(path.join(__dirname,'../gaussian_box1_time20_v1_20261009/pack_preview.cjs'));
